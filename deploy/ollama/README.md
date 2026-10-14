# Ollama CPU 폴백 장애 기록 (2026-10-06 ~ 10-09)

**이 디렉터리에 있던 systemd 가드는 제거됐다.** 가드가 감시하던 대상(`brbs-etl` 호스트의
`brbs-ollama` docker 컨테이너)이 더 이상 없다 — ollama는 `soma-k8s` 클러스터의 파드가 됐다.
남겨둔 이유는 자동화가 아니라 **장애의 성질을 기록해 두는 것**이다. 아래 실패 모드는
docker의 성질이 아니라 Ollama의 성질이라, 서버가 어디서 돌든 같은 방식으로 재현된다.

## 무슨 일이 있었나

VLM 호출 228회가 전부 600초 타임아웃으로 깨졌다(누적 6,324회 중 3.6%). 그 끝에 호스트
`brbs-etl`이 OOM으로 죽어 6시간 방치됐다.

원인은 Ollama의 **조용한 CPU 폴백**이다. 기동 시 GPU 디스커버리에 실패해도 에러를 내지 않는다.

```text
level=INFO msg="failure during llama-server GPU discovery" error="context deadline exceeded"
level=INFO msg="inference compute" id=cpu library=cpu total="15.4 GiB"
level=INFO msg="vram-based default context" total_vram="0 B" default_num_ctx=4096
```

정상이면 이렇게 찍힌다.

```text
level=INFO msg="inference compute" id=0 library=CUDA compute=7.5 name="Tesla T4" total="14.6 GiB"
```

CPU 모드의 qwen3-vl:8b는 페이지 한 장에 600초를 넘기고(정상은 20~40초), 가중치 12.6GiB를
호스트 RAM에 올린다. 15.4GiB짜리 호스트에서 그건 OOM이다. 스왑이 없어 완충도 없다.

전말은 [GPU 트러블 슈팅](https://app.notion.com/p/3f430798f08f807d8a19dd0cae1d8255) 참고.

## 왜 헬스체크로는 안 잡혔나

컨테이너에 헬스체크가 **이미 있었다.** 그런데 이것이다.

```
ollama list >/dev/null 2>&1 || exit 1
```

CPU 모드에서도 멀쩡히 통과한다. API는 살아 있기 때문이다. 이 장애가 끝까지 안 보인 이유다.
그리고 docker는 **unhealthy를 이유로 컨테이너를 재시작하지 않는다**(그건 Swarm 기능).
헬스체크를 고쳐도 그 자체로는 아무것도 복구되지 않아서, 가드를 바깥 systemd 타이머에 뒀다.

가드의 판정은 "언제 떨어졌나"가 아니라 "지금 떨어져 있나"였다. 관측된 계기는 부팅 직후
드라이버가 준비되기 전에 컨테이너가 뜨는 레이스였지만, OOM으로 러너가 죽고 Ollama가 다시
뜨는 경로에서도 똑같이 CPU로 떨어지기 때문이다.

## k8s로 옮기면서 무엇이 바뀌었나

**관측된 계기(부팅 레이스)는 구조적으로 사라졌다.** ollama 파드가 `nvidia.com/gpu: 1`을
요청하고, device plugin이 그 자원을 광고하지 않으면 파드는 스케줄되지 않는다. 드라이버가
준비되지 않은 노드에 파드가 뜨는 일 자체가 없다.

**남아 있는 경로는 "뜬 뒤에 드라이버가 죽는" 쪽이다.** device plugin이 디바이스를 unhealthy로
신고하면 kubelet이 파드를 쳐내지만, time-slicing(물리 GPU 1장을 2개로 광고) 상태에서 그게
실제로 동작하는지는 확인되지 않았다. 그리고 지금은 가드도 없다.

그래서 등가의 신호를 아직 세우지 않은 상태다. 가장 싼 대체는 클러스터 Alloy가 이미
`/var/log/pods`를 걷고 있으므로 **ollama 로그의 `library=cpu`에 Loki 알림을 거는 것**이다.

## 지금 상태를 직접 확인하는 방법

```bash
# 모델이 실제로 VRAM에 있는지. gpu_pct가 100이 아니면 CPU로 떨어진 것이다.
kubectl exec deploy/ollama -- \
  sh -c "curl -s localhost:11434/api/ps" | jq '.models[] | {name, gpu_pct: (.size_vram / .size * 100)}'

# 현재 기동분의 디바이스 판정
kubectl logs deploy/ollama | grep "inference compute"
```

`models`가 비어 있으면(모델이 아직 안 올라감) 로그 판정이 유일한 근거다.

## 예방은 어디까지 되나 — 당시 검토

가드는 사후 복구였다. 애초에 CPU 모드로 뜨지 않게 하는 쪽은 아래와 같았고, 어느 것도
단독으로 완전하지 않았다. `nvidia.com/gpu` 요청이 첫 줄을 대체했다고 보면 된다.

| 방법 | 효과 | 한계 |
|---|---|---|
| 엔트리포인트 GPU 대기 | "GPU가 실제로 보일 때"를 직접 기다린다 | nvidia-smi가 보이는 것과 Ollama 디스커버리 성공은 다르다 |
| 기동 순서 고정 | 부팅 레이스를 줄인다 | `After=`는 "유닛 종료"지 "GPU 준비"가 아니다 |
| 퍼시스턴스 모드 | 드라이버 초기화 지연을 없앤다 | **이미 켜져 있었다**(nvidia-smi 0.02s). 더 할 게 없다 |
| 디스커버리 타임아웃 상향 | 워치독이 안 터지게 | **불가.** Ollama가 환경변수로 노출하지 않는다 |
| 호스트 부하 낮추기 | 워치독이 터질 확률을 낮춘다 | 간접적. `INGEST_CONCURRENCY=1` 유지 |

부팅 레이스가 얼마나 빡빡했는지: 2026-10-09 재부팅에서 **부팅 07:32:24 → ollama 기동
07:32:36**, 12초다.

## 덤으로 남는 교훈

- **로그 로테이션이 없으면 진단 도구가 부하가 된다.** 당시 `brbs-ollama`의 json 로그는
  633MB / 430만 줄까지 자라 있었고(2026-10-09 실측), `docker logs --since`가 파일 앞에서부터
  훑기 때문에 한 번 읽는 데 11.49초가 걸렸다. 1분 타이머가 그걸 반복하면 가드 자신이
  호스트 부하가 된다. 그래서 가드는 기동당 1회만 판정하고 결과를 캐시했다(적중 시 0.04s).
- **디바이스 결정은 기동 때 한 번 일어나고 그 뒤로 바뀌지 않는다.** 그래서 기동 단위 캐시가
  성립했고, 로그가 로테이션돼 기동 줄이 잘려도 판정을 잃지 않았다.
- **VLM 쪽 방어는 코드에 남아 있다.** `VLM_FAIL_STREAK`가 연속 실패를 문서 단위로 접는다.
  이게 없을 때 214페이지 문서 하나로 35.6시간을 태웠다.
