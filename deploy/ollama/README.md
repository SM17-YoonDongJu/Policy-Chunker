# brbs-ollama GPU 가드 — CPU 폴백을 막는다

`deploy/observability/`와 같은 성격이다. **CD가 배포하지 않는다.** `deploy.yml`이 SSM으로
호스트에 보내는 건 `deploy/docker-compose.prod.yml`과 `deploy/remote-deploy.sh` 둘뿐이고,
brbs-ollama는 이 레포의 배포 대상이 아니다. 여기 두는 목적은 자동화가 아니라 **재현**이다.

## 무슨 일이 있었나

2026-10-06 ~ 10-09, VLM 호출 228회가 전부 600초 타임아웃으로 깨졌다(누적 6,324회 중 3.6%).
그 끝에 호스트 `brbs-etl`이 OOM으로 죽어 6시간 방치됐다.

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

## 왜 헬스체크로는 안 잡히나

brbs-ollama에 헬스체크가 **이미 있다.** 그런데 이것이다.

```
ollama list >/dev/null 2>&1 || exit 1
```

CPU 모드에서도 멀쩡히 통과한다. API는 살아 있기 때문이다. 이 장애가 끝까지 안 보인 이유다.

그리고 docker는 **unhealthy를 이유로 컨테이너를 재시작하지 않는다**(그건 Swarm 기능).
헬스체크를 고쳐도 그 자체로는 아무것도 복구되지 않는다. 그래서 가드를 밖에 둔다.

## 무엇을 하나

systemd 타이머가 1분마다 "지금 CPU 모드인가"를 보고, 맞으면 컨테이너를 재시작한다.

| 판정 | 근거 | 동작 |
|---|---|---|
| 정상 | 현재 기동분 로그에 `library=CUDA` | 카운터 초기화 |
| CPU 모드 | `inference compute` 줄이 있는데 전부 `library=cpu` | 재시작 (최대 3회) |
| 기동 중 | `inference compute` 줄이 아직 없음 (120초 유예) | 판정 보류 |
| 상한 초과 | 3회 재시작해도 CPU 모드 | 중단하고 systemd에 실패로 남김 |

**기동 계기를 따지지 않는 게 핵심이다.** 관측된 계기는 부팅 직후 드라이버가 준비되기 전에
컨테이너가 뜨는 레이스였지만, OOM으로 러너가 죽고 Ollama가 다시 뜨는 경로에서도 똑같이
CPU로 떨어진다. "언제 떨어졌나"가 아니라 "지금 떨어져 있나"를 본다.

재시작 상한을 두는 이유: 드라이버가 실제로 죽었거나 다른 프로세스가 GPU를 쥐고 있으면
재시작으로는 안 고쳐진다. 그때 무한 재시작은 아무것도 아닌 것보다 나쁘다.

## 적용 (brbs-etl 호스트)

```bash
sudo install -m 755 ollama-gpu-guard.sh /usr/local/bin/ollama-gpu-guard.sh
sudo install -m 644 ollama-gpu-guard.service /etc/systemd/system/
sudo install -m 644 ollama-gpu-guard.timer   /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ollama-gpu-guard.timer
```

## 확인

```bash
# 한 번 수동 실행 — 정상이면 아무 일도 안 일어난다
sudo /usr/local/bin/ollama-gpu-guard.sh

# 타이머가 도는지
systemctl list-timers ollama-gpu-guard.timer

# 판정 이력
journalctl -t ollama-gpu-guard -n 20

# 지금 모델이 실제로 VRAM에 있는지 (가드와 별개로 직접 확인)
curl -s localhost:11434/api/ps | jq '.models[] | {name, gpu_pct: (.size_vram / .size * 100)}'
```

마지막 명령이 `gpu_pct: 100`이 아니면 CPU에 떨어진 것이다. 모델이 안 올라가 있으면
`models`가 비어 있고, 그때는 가드의 로그 판정이 유일한 근거다.

## 되돌리기

```bash
sudo systemctl disable --now ollama-gpu-guard.timer
sudo rm /etc/systemd/system/ollama-gpu-guard.{service,timer} /usr/local/bin/ollama-gpu-guard.sh
sudo systemctl daemon-reload
```

## 판정을 캐시하는 이유

디바이스 결정은 기동할 때 한 번 일어나고 그 뒤로 바뀌지 않는다. 그래서 한 기동을 GPU로
확인하면 `StartedAt`과 함께 기록해 두고, 그 기동 동안은 로그를 다시 읽지 않는다.

성능 때문이다. brbs-ollama의 json 로그는 로테이션 설정이 없어 **633MB / 430만 줄**까지
자라 있었고(2026-10-09 실측), `docker logs --since`는 파일 앞에서부터 훑기 때문에 한 번
읽는 데 11.49초가 걸린다. 1분 타이머가 그걸 매번 반복하면 가드 자신이 호스트 부하가 된다.

| | 소요 |
|---|---|
| 캐시 없음 (633MB 스캔) | 11.49s |
| 캐시 적중 | 0.04s |

덤으로 로그가 로테이션돼 기동 줄이 잘려나가도 판정을 잃지 않는다.

## 예방은 어디까지 되나

가드는 **사후 복구**다. 애초에 CPU 모드로 뜨지 않게 하는 쪽은 아래와 같은데, 어느 것도
단독으로 완전하지 않다.

| 방법 | 효과 | 한계 |
|---|---|---|
| 엔트리포인트 GPU 대기 | "GPU가 실제로 보일 때"를 직접 기다린다. 모든 기동 경로에 적용 | 컨테이너 재생성 필요. nvidia-smi가 보이는 것과 Ollama 디스커버리가 성공하는 건 다르다 |
| docker 기동 순서 고정 | 부팅 레이스를 줄인다 | docker 전체가 늦어져 다른 컨테이너도 밀린다. `After=`는 "유닛 종료"지 "GPU 준비"가 아니다 |
| 퍼시스턴스 모드 | 드라이버 초기화 지연을 없앤다 | **이미 켜져 있다**(`persistence_mode: Enabled`, nvidia-smi 0.02s). 더 할 게 없다 |
| 디스커버리 타임아웃 상향 | 워치독이 안 터지게 | **불가.** Ollama가 환경변수로 노출하지 않는다(`OLLAMA_LOAD_TIMEOUT`은 모델 적재용) |
| 호스트 부하 낮추기 | 워치독이 터질 확률을 낮춘다 | 간접적. `INGEST_CONCURRENCY=1` 유지, 로그 로테이션 |

부팅 레이스가 얼마나 빡빡한지: 2026-10-09 재부팅에서 **부팅 07:32:24 → ollama 기동
07:32:36**, 12초다. `docker.service`는 `After=firewalld.service`뿐이고 nvidia 관련 유닛과
순서 관계가 아예 없다.

결론: 예방은 **빈도**를 낮추고, 가드는 **피해**를 묶는다. 둘 중 하나만 고른다면 가드다.
예방이 뚫렸을 때 가드가 없으면 38시간이고, 있으면 1분이다.

## 다음에 컨테이너를 다시 만들 때

같이 넣을 것 세 가지. 지금 안 한 이유는 컨테이너 재생성이 필요해서다
(brbs-corpus-worker도 같은 Ollama를 쓰므로 잠깐이라도 끊기면 그쪽에 영향이 간다).

**1. 엔트리포인트 GPU 대기**

```bash
until nvidia-smi -L >/dev/null 2>&1; do sleep 2; done
exec /bin/ollama serve
```

**2. 로그 로테이션** — 지금 설정이 없어 633MB까지 자랐다. 디스크가 차면 호스트가 또 죽는다.
우리 쪽 `docker-compose.prod.yml`과 같은 값을 쓴다.

```
--log-opt max-size=10m --log-opt max-file=3
```

**3. 헬스체크 교체** — 모델이 올라와 있는데 VRAM이 0이면 unhealthy로 본다. 지금 것은
`ollama list`라 CPU 모드에서도 통과한다. docker가 그걸로 재시작해 주지는 않으므로
(그건 Swarm 기능) 가드는 그대로 둔다.

## 현재 컨테이너 사양 (2026-10-09 기준)

수기 `docker run`으로 떠 있고 compose 라벨이 없다. 어디에도 기록이 없어서 여기 남긴다.

| 항목 | 값 |
|---|---|
| 이미지 | `ollama/ollama:latest` |
| 엔트리포인트 | `/bin/ollama serve` |
| 포트 | `127.0.0.1:11434 -> 11434/tcp` (bridge) |
| 볼륨 | `ollama-models` -> `/root/.ollama` |
| GPU | `--gpus all` (DeviceRequests count=-1) |
| 환경변수 | `NVIDIA_VISIBLE_DEVICES=all`, `NVIDIA_DRIVER_CAPABILITIES=compute,utility`, `OLLAMA_HOST=0.0.0.0:11434` |
| 재시작 | `unless-stopped` |
| 모델 | `qwen3-vl:8b-instruct`(6.1GB), `qwen3-embedding:0.6b`(0.6GB) |

참고로 `deploy/.env.example`과 README가 "ollama도 network_mode: host"라고 적고 있는데
사실이 아니다. bridge에 `127.0.0.1:11434`를 퍼블리시한 구조다. 우리 컨테이너가
`network_mode: host`라서 `localhost:11434`로 닿는 결과는 같지만, 트러블슈팅에서 엉뚱한
데를 보게 되므로 고쳐 두는 게 좋다.
