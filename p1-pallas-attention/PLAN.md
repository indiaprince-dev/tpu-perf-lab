# P1 — Pallas TPU 커널 + Roofline 분석 (실행계획)

> 투입: **주 25~30시간** (평일 3~4h × 5 + 주말 8~10h)
> 기간: **6주** · 착수 2026.08 중순 → **완료 2026.09 말**
> 언어: **Python (JAX/Pallas)** — C++ 불필요

---

## 목표 산출물

1. 공개 GitHub 레포 (재현 가능한 코드 + 테스트)
2. README: **roofline 차트 · 오토튜닝 히트맵 · before/after 성능표**
3. 기술 블로그 1편 (영문 권장 — 링크드인·SOP 양쪽에 사용)
4. **SOP·이력서에 넣을 한 문장**: "roofline 분석으로 memory-bound 구간을 식별하고 Pallas 커널을 재작성해 N배 속도 향상"

---

## 타깃 연산 선정

| 후보 | 난이도 | 임팩트 | 판단 |
|---|---|---|---|
| Elementwise fusion (LayerNorm 등) | ★ | 낮음 | **Week 3 연습용** |
| **Fused Attention (Flash 스타일)** | ★★★★ | **높음** | ⭐ **본 타깃** |
| Quantized matmul (int8) | ★★★ | 높음 | 대안 (실패 시 전환) |
| Top-k / sampling 커널 | ★★ | 중간 | 백업 |

**전략**: Week 3에 elementwise fusion으로 Pallas 문법을 익히고, Week 4에 fused attention으로 확장.
attention이 막히면 quantized matmul로 전환 (Week 4 중반에 판단).

> attention을 고르는 이유: TPU Perf 팀의 실제 워크로드(Gemini 등 LLM)의 핵심 병목이고,
> 면접에서 "왜 이 연산을 골랐나"에 답이 명확합니다.

---

## 주차별 계획

### Week 1 — 환경 + TPU 멘탈모델 (25h)
| 항목 | 시간 |
|---|---|
| TRC 신청 결과 확인 / Colab·Kaggle TPU 백업 셋업 | 3h |
| 『How to Scale Your Model』 Ch1-2 (TPU 하드웨어, JAX 프로그래밍) | 8h |
| JAX 기초: `jit`, `vmap`, `shard_map`, 메모리 계층 | 8h |
| **matmul 벤치마크** — 이론 peak FLOPS 대비 실측 | 6h |

✅ **완료 기준**: TPU에서 matmul을 돌리고 "이론 대비 몇 %"를 숫자로 말할 수 있다.

### Week 2 — 프로파일링 + Roofline (25h)
| 항목 | 시간 |
|---|---|
| XProf / `jax.profiler.start_trace` 사용법 | 6h |
| scaling-book "How to Profile TPU Programs" 장 | 5h |
| **베이스라인 attention 구현** (순수 JAX) + 트레이스 획득 | 8h |
| **Roofline 계산**: arithmetic intensity = FLOPs / HBM bytes → memory/compute-bound 판정 | 6h |

✅ **완료 기준**: 베이스라인의 roofline 차트가 나오고, 병목이 어디인지 근거를 대고 말할 수 있다.

### Week 3 — Pallas 문법 습득 (25h)
| 항목 | 시간 |
|---|---|
| JAX Pallas TPU 문서 정독 (`BlockSpec`, `grid`, HBM→VMEM) | 8h |
| **연습 커널**: elementwise fusion (LayerNorm) 작성 | 10h |
| 정확성 테스트 작성 (reference 대비 allclose) | 4h |
| 성능 측정 → XLA 기본 구현과 비교 | 3h |

✅ **완료 기준**: 동작하는 Pallas 커널 1개 + 통과하는 테스트.

### Week 4 — 본 타깃 커널 v1 (28h)
| 항목 | 시간 |
|---|---|
| Fused attention 커널 설계 (타일링 전략 결정) | 6h |
| 커널 v1 구현 | 14h |
| 정확성 검증 + 디버깅 | 8h |

⚠️ **Week 4 중반 판단 포인트**: 진행이 막히면 **quantized matmul로 전환**. 6주 안에 완성되는 것이 우선.

✅ **완료 기준**: 수치적으로 정확한 커널 v1.

### Week 5 — 최적화 + 오토튜닝 (28h)
| 항목 | 시간 |
|---|---|
| 커널 v2: 파이프라이닝, 컴퓨트/HBM 전송 오버랩 | 10h |
| **블록 사이즈 스윕** → 성능 히트맵 생성 | 8h |
| XProf 커널 프로파일링으로 개선 원인 규명 | 6h |
| before/after 성능표 정리 | 4h |

✅ **완료 기준**: 히트맵 + "왜 이 설정이 빠른가"에 대한 하드웨어 수준 설명.

### Week 6 — 문서화 (20h)
| 항목 | 시간 |
|---|---|
| README 작성 (차트·표·재현 방법) | 6h |
| 기술 블로그 1편 (영문) | 10h |
| 코드 정리, 재현성 검증 (clean clone → 실행) | 4h |

✅ **완료 기준**: 남이 clone해서 돌릴 수 있고, 읽으면 무엇을 왜 했는지 알 수 있다.

---

## 리스크 & 대응

| 리스크 | 대응 |
|---|---|
| TRC 승인 지연 | Colab/Kaggle TPU로 시작 (v5e 접근 가능). TRC는 Week 3부터 필요 |
| Pallas 학습곡선 | Week 3 연습 커널을 반드시 거칠 것. 바로 attention 가면 막힘 |
| Attention 커널 실패 | **Week 4 중반 전환 판단** → quantized matmul |
| 성능 개선이 안 나옴 | **"개선 실패"도 유효한 결과.** 왜 안 되는지 roofline으로 설명하면 오히려 깊이 있는 포트폴리오 |
| KAIST 3학기 + 현업과 충돌 | 주말 블록을 Week 4~5(구현 집중)에 배치 |

> 💡 **성능 수치가 안 나와도 프로젝트는 성립합니다.**
> 면접에서 평가되는 건 "몇 배 빨라졌나"가 아니라 **"병목을 어떻게 규명했나"** 입니다.

---

## 체크리스트

- [ ] TRC 신청 제출
- [ ] Colab TPU에서 JAX hello world
- [ ] GitHub 레포 생성 (public)
- [ ] scaling-book Ch1-2 완독
- [ ] matmul 벤치마크 (이론 대비 %)
- [ ] XProf 트레이스 획득
- [ ] Roofline 차트 (베이스라인)
- [ ] 연습 커널 (LayerNorm) 동작
- [ ] 타깃 커널 v1 정확성 통과
- [ ] 커널 v2 최적화
- [ ] 오토튜닝 히트맵
- [ ] README + before/after 표
- [ ] 블로그 1편
- [ ] 재현성 검증
