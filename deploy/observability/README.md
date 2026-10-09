# 관측 스택 연결 — Alloy(로그) · Prometheus 스크랩(메트릭)

이 디렉터리는 **CD가 배포하지 않는다.** `deploy.yml`이 SSM으로 호스트에 보내는 건
`deploy/docker-compose.prod.yml`과 `deploy/remote-deploy.sh` 둘뿐이고, 실행하는 것도
`docker compose up -d --no-deps chunker`다. 여기 있는 건 다른 compose 프로젝트라 시야 밖이다.

여기 두는 목적은 자동화가 아니라 재현이다. 이 설정은 2026-08-31에 손으로 올렸고, 그때까지
어디에도 기록돼 있지 않아서 인스턴스를 다시 만들면 통째로 사라지는 상태였다.

## 왜 필요했나

`/metrics`는 데몬이 스스로 열고(`exporter.py`), 로그는 JSON으로 찍는다(`logging_setup.py`).
둘 다 "누가 가져가는지"는 이 레포 밖의 일인데, 그 연결이 실제로는 없었다.

- brbs-etl 호스트에 Alloy가 없어서 로그가 Loki에 들어가지 않았다
- Prometheus는 이 호스트의 cadvisor(8080)·node exporter(9100)만 긁고 있었고 9101은 대상이 아니었다
- 배포 검증(`remote-deploy.sh`의 `report()`)이 `/metrics`를 curl 해보긴 하지만 응답이 없어도
  실패로 치지 않는다 — 그래서 이 구멍이 배포 초록불 뒤에 계속 숨어 있었다

## 구성

```
brbs-etl (10.0.11.131)                 brbs-monitoring (10.0.11.48)
  brbs-insurance-chunker  ─stdout→
  brbs-alloy  ─────────────push───────→  brbs-loki:3100
                          ←──scrape────  brbs-prometheus  (:9101)
```

Alloy는 `/var/run/docker.sock`으로 같은 호스트의 컨테이너를 발견해 로그를 읽는다. 원격
컨테이너의 stdout은 읽을 수 없으므로 **로그를 만드는 호스트마다 하나씩** 있어야 한다.

## 보안그룹

기본 allow-all egress가 아니므로 나가는 쪽도 명시해야 한다(실제로 여기서 한 번 막혔다 —
인바운드만 열고 아웃바운드를 안 열어 타임아웃이 났다).

| 방향 | 대상 SG | 규칙 |
|---|---|---|
| Alloy → Loki (인바운드) | `sg-0ac416019c28c8679` monitoring | TCP 3100 ← `sg-085aebc768844b06c` |
| Alloy → Loki (아웃바운드) | `sg-085aebc768844b06c` etl | TCP 3100 → `sg-0ac416019c28c8679` |
| Prometheus → /metrics | `sg-085aebc768844b06c` etl | TCP 9101 ← `sg-0ac416019c28c8679` |

## 적용 (brbs-etl 호스트)

`~/insurance-chunker/`가 **아닌** 곳에 둔다. 그 디렉터리의 compose는 배포가 매번 덮어쓴다.

```bash
mkdir -p ~/observability && cd ~/observability
# 이 디렉터리의 docker-compose.yml, config.alloy를 여기로 복사
docker compose up -d
```

확인:

```bash
docker logs brbs-alloy --tail 40                                  # 설정 파싱 에러
curl -s localhost:12345/metrics | grep -E 'loki_write_(sent|dropped)'
curl -s localhost:12345/metrics | grep loki_source_docker
```

`loki_source_docker_target_entries_total`과 `loki_write_sent_entries_total`이 같고
`dropped`가 0이면 끝에서 끝까지 붙은 것이다.

## Prometheus 스크랩 (brbs-monitoring 호스트)

Prometheus가 `127.0.0.1:9090`에만 묶여 있어 remote_write로 밀어넣는 경로는 쓸 수 없다.
pull로 간다. 설정 파일 위치는 이렇게 찾는다.

```bash
sudo docker inspect brbs-prometheus \
  --format '{{range .Mounts}}{{.Source}} -> {{.Destination}}{{"\n"}}{{end}}'
```

`scrape_configs`에 추가하고 재시작한다.

```yaml
  - job_name: insurance-chunker
    scrape_interval: 30s
    static_configs:
      - targets: ['10.0.11.131:9101']
        labels:
          service: insurance-chunker
```

## 알림을 걸 때

주기가 7일(`INGEST_INTERVAL_SECONDS=604800`)이라 대부분의 시간 동안 카운터가 안 움직인다.
활동량 기반 알림(`rate()`, "5분간 처리 0건")은 상시 발화한다. 주 신호는 신선도 게이지다.

```promql
# 마지막 성공이 한 주기 반을 넘었다
time() - insurance_chunker_last_success_timestamp_seconds > 604800 * 1.5

# 재시도 상한에 걸려 이번 주기에 손도 안 댄 문서
insurance_chunker_quarantined_documents > 0

# 적재는 됐지만 경계를 못 잡아 조번호가 어긋난 문서
insurance_chunker_weak_boundary_documents > 0
```

`insurance_chunker_last_success_timestamp_seconds`는 **값이 있을 때만 나온다.** 지표가 아예
안 보이면 "오래 실패했다"가 아니라 "완료된 사이클이 한 번도 없다"는 뜻이다 — `absent()`로
따로 잡아야 한다.

## 로그 질의

```logql
{job="brbs/insurance-chunker", level="ERROR"}
{job="brbs/insurance-chunker", event="boundary_weak"} | json
sum by (status) (
  count_over_time({job="brbs/insurance-chunker", event="document_done"} | json [24h])
)
```

`level`·`service`·`event`만 라벨이고 나머지(`document`, `sha256`, `chunks` 등)는 본문이라
`| json` 이후에 필터해야 한다. 라벨 카디널리티 정책상 의도된 것이다.
