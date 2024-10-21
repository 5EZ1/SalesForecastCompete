<<<<<<< HEAD
# SalesForecastCompete
=======
# SalesForecastCompete
=======
◦ 분석 개요&#x20;

ㅇ 분석 목적 및 분석과정 (데이터 분류 과정 작성 필수)

이번 분석의 목적은 유통데이터를 활용하여 특정 상품군의 수요를 예측함으로써 보다 효율적인 재고관리 및 공급 체계를 구축하는 것입니다. 향후 6개월의 수요를 예측하고, 주말과 공휴일 등의 외부 요인도 고려하여 정확한 예측 모델을 개발하는 것을 목표로 합니다.

본 프로젝트는 SARIMA와 XGBoost 모델을 이용한 비교 분석을 통해 최적의 수요 예측 모델을 선정하는 것을 주된 목표로 삼고 있습니다. 이를 위해 다양한 전처리 과정과 통계적 분석을 거쳐 모델을 구축하였으며, 분석 과정을 통해 도출된 인사이트를 활용하여 실제 운영에 적용할 수 있는 방안을 제시합니다.



◦ 분석 방법 및 절차&#x20;

ㅇ 분석모형에 대한 서술

ㅇ 구체적인 분석 방법 및 절차에 대해 서술&#x20;

1. **데이터 수집 및 전처리**

   - 데이터는 (1 데이터)와 (2 데이터)로 구분되며, 각각 특정 중분류와 대분류의 수요 예측을 목표로 합니다. 데이터의 특성상 결측치 처리, 이상치 제거, 중복 제거 등의 전처리 작업을 거쳤습니다. 특히 '반품' 데이터는 분석 대상에서 제외하고, 판매 수량의 로그 변환 및 다양한 스케일링 기법을 적용하여 데이터의 정상성을 확보했습니다.

2. **시계열 분석 및 모델 선정**

   - SARIMA 모델을 사용하여 수요 예측을 수행했습니다. SARIMA는 계절성을 가진 시계열 데이터에 적합하며, 판매 데이터의 월별 계절성을 반영하기 위해 SARIMA(1, 1, 1) x (1, 1, 1, 12) 모델을 적용하였습니다. 또한 모델링 단계에서 공휴일 및 주말 여부와 같은 외생 변수를 추가하여 예측 성능을 향상시키기 위해 노력했습니다.

3. **모델 학습 및 평가**

   - 학습된 SARIMA 모델을 사용하여 2024년 1월부터 6월까지의 판매 수요를 예측하였습니다. 모델의 적합도는 AIC, BIC 등의 기준을 통해 평가하였으며, RMSE 지표를 사용해 예측 성능을 측정하였습니다. 예측된 값과 실제 값 간의 비교를 통해 모델의 성능을 검토하였습니다.

4. **인사이트 도출 및 적용 방안**

   - 분석 결과를 바탕으로 도출된 인사이트를 통해, 특정 시점(예: 명절 등)에서의 수요 증가를 반영한 재고 관리 전략을 제안하였습니다. 또한, 주말과 공휴일의 판매 패턴을 반영하여 매장 운영 계획을 최적화할 수 있는 방안을 모색하였습니다. 실제 운영에서 이를 적용하여 불필요한 재고 비용을 줄이고, 수요에 따른 최적의 재고 수준을 유지할 수 있도록 하였습니다.

◦ 분석 결과

ㅇ 분석한 결과에 대해 서술&#x20;

이번 분석에서는 SARIMA 모델을 이용해 중분류와 대분류의 수요를 예측하였습니다. SARIMA 모델을 사용하여 계절성을 반영한 시계열 분석을 수행했으며, 공휴일 및 주말과 같은 외부 요인을 반영하여 예측 성능을 향상시켰습니다.

- **SARIMA 모델의 성능**: SARIMA 모델의 예측 성능을 RMSE, AIC, BIC 등의 지표로 평가하였으며, RMSE는 6,000\~7,000 수준으로 나타났습니다. 이는 모델이 일정 수준의 예측 정확성을 가지지만, 일부 외부 요인에 따라 오차가 발생할 수 있음을 의미합니다.
- **수요 패턴의 주요 특징**: 분석 결과에 따르면, 명절과 같은 특정 시점에서 수요가 급격히 증가하는 패턴이 관찰되었습니다. 이러한 패턴을 반영하여 재고를 관리함으로써 수요 변화에 유연하게 대응할 수 있는 방안을 마련할 수 있었습니다.
- **주말 및 공휴일의 영향**: 주말과 공휴일의 경우, 일반적인 평일에 비해 판매량이 감소하는 경향이 있었습니다. 이를 통해 매장 운영 시간을 조정하거나 재고 관리 전략을 수정하는 등 보다 효율적인 운영 방안을 모색할 수 있었습니다.

또한, SARIMA 모델을 사용하여 향후 6개월의 수요를 예측한 결과, 월별 수요의 변동성을 파악하고 예측된 수요를 기반으로 한 재고 관리 전략 수립에 기여할 수 있음을 확인하였습니다. 이를 통해 실제 운영에 적용 가능한 인사이트를 제공하고, 재고 비용 절감 및 매출 극대화를 위한 전략적 의사결정에 도움을 줄 수 있었습니다.

**◦ 활용방안**

ㅇ 제시한 모델의 활용방안 및 향후 연구에 대한 방향성에 대해 서술

본 분석 결과를 통해 도출된 인사이트는 다음과 같은 실제 운영 방안에 활용될 수 있습니다:

1. **효율적인 재고 관리**: 예측된 수요를 바탕으로 적정 재고 수준을 유지함으로써 불필요한 재고 비용을 줄일 수 있습니다. 특히, 명절과 같은 특정 시점에서 수요가 급증하는 패턴을 반영하여 선제적인 재고 확보가 가능하며, 이를 통해 품절 리스크를 최소화할 수 있습니다.

2. **운영 계획 최적화**: 주말 및 공휴일의 판매량 감소 패턴을 반영하여 매장 운영 시간을 효율적으로 조정할 수 있습니다. 이를 통해 인건비와 기타 운영 비용을 절감하고, 효율적인 인력 배치를 통해 매출을 극대화할 수 있는 기회를 제공합니다.

3. **프로모션 및 마케팅 전략 수립**: 수요 예측 데이터를 활용하여 특정 시점에서의 프로모션을 기획할 수 있습니다. 예를 들어, 명절 직전 수요가 증가하는 시점을 타겟으로 한 할인 행사나 마케팅 캠페인을 통해 매출을 증대시킬 수 있습니다.

4. **물류 최적화**: 예측된 수요 데이터를 활용하여 물류 및 배송 계획을 최적화할 수 있습니다. 이를 통해 배송 지연을 방지하고, 물류 비용을 절감하는 데 기여할 수 있습니다.



◦ 활용데이터 및 참고 문헌 출처 등

ㅇ 활용 데이터명 : (1 데이터), (2 데이터)

ㅇ 타기관 또는 민간 데이터 명 : 없음

ㅇ 참고한 관련 문헌이 있을 경우 작성: 본 분석에서는 제공된 유통데이터 외에 추가적인 외부 데이터를 사용하지 않았습니다.







>>>>>>> 119f23a (Add_README)
>>>>>>> ad7bf98 (Initial commit)
