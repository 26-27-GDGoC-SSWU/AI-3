# 3~5장: 분류, 모델 훈련, 서포트 벡터 머신
핸즈온 머신러닝 | 날짜: 2026-09-14 | 작성자: 강예담

---

# 3장. 분류 (Classification)

## 1. MNIST
MNIST는 손글씨 숫자(0~9) 이미지 7만 개짜리 데이터셋. 각 이미지는 28×28 픽셀이라 숫자 784개짜리 리스트로 표현된다.

```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', as_frame=False)

X, y = mnist.data, mnist.target
# X: 이미지 데이터 (70000개 행, 784개 열)
# y: 정답 라벨 (실제 숫자 0~9)

X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]
# 앞 6만 개는 훈련용, 뒤 1만 개는 테스트용으로 미리 나눠져 있음
```

## 2. 이진 분류기 훈련
"이 이미지가 5인가, 아닌가?"만 판단하는 모델을 만든다.

```python
from sklearn.linear_model import SGDClassifier

y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')

sgd_clf = SGDClassifier(random_state=42)
# random_state=42: 결과를 재현 가능하게 고정하는 값

sgd_clf.fit(X_train, y_train_5)
sgd_clf.predict([X_train[0]])
```

## 3. 성능 측정
정확도만 보면 위험하다 (5가 데이터의 10%뿐이면, 무조건 "5 아님"이라 해도 정확도 90%). 그래서 여러 지표를 같이 본다.

```python
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
cm = confusion_matrix(y_train_5, y_train_pred)
print(cm)
# [[TN, FP], [FN, TP]] 형태로 출력됨

from sklearn.metrics import precision_score, recall_score, f1_score
precision_score(y_train_5, y_train_pred)  # 정밀도
recall_score(y_train_5, y_train_pred)     # 재현율
f1_score(y_train_5, y_train_pred)         # F1 점수 (정밀도·재현율의 조화평균)
```

```python
from sklearn.metrics import precision_recall_curve, roc_curve

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
# matplotlib으로 그래프 그려서 시각적으로 비교 가능
```

## 4. 다중 분류
0~9 열 개 숫자를 다 구분해야 하는 문제.

```python
from sklearn.svm import SVC

svm_clf = SVC(random_state=42)
svm_clf.fit(X_train[:2000], y_train[:2000])
# 사이킷런이 자동으로 OvO(One-vs-One) 방식을 내부적으로 씀

svm_clf.predict([X_train[0]])
```

```python
from sklearn.multiclass import OneVsRestClassifier

ovr_clf = OneVsRestClassifier(SVC(random_state=42))
ovr_clf.fit(X_train[:2000], y_train[:2000])
# 명시적으로 OvR(One-vs-Rest) 방식을 쓰고 싶을 때 이렇게 감싸줌
```

## 5. 오류 분석
모델이 어떤 숫자를 어떤 숫자로 자주 헷갈리는지 오차 행렬을 이미지처럼 그려서 확인한다.

```python
from sklearn.metrics import ConfusionMatrixDisplay

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, normalize="true")
```

## 6. 다중 레이블 분류
한 이미지에 정답이 여러 개일 수 있는 경우. 예: "7보다 큰가?" + "홀수인가?" 두 질문에 동시에 답.

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train.astype('int8') >= 7)
y_train_odd = (y_train.astype('int8') % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
knn_clf.predict([X_train[0]])
# 출력 예: [False, True] → "7보다 작고, 홀수다"
```

## 7. 다중 출력 분류
다중 레이블을 확장해서, 각 레이블이 여러 값(0~255 같은)을 가질 수 있는 경우.

```python
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
y_train_mod = X_train

knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_train_mod[0]])
# 노이즈 낀 이미지를 넣으면 깨끗한 이미지를 예측해서 출력
```

## 퀴즈 / 몰랐던 부분
**Q.** 데이터의 90%가 "정상", 10%가 "이상"인 불균형 데이터에서, 정확도만 보고 모델 성능을 판단하면 안 되는 이유는?

**A.** 무조건 "정상"이라고만 예측해도 정확도 90%가 나오기 때문 (정확도의 함정). 대신 **오차 행렬**, **정밀도**, **재현율**, 그리고 이를 종합한 **F1 점수**를 함께 봐야 한다.

---

# 4장. 모델 훈련 (Training Models)

## 1. 선형 회귀
데이터에 가장 잘 맞는 직선(또는 초평면)을 찾는 것: `y = θ₀ + θ₁x₁ + θ₂x₂ + ...`. **정규방정식**은 미분으로 최적의 θ를 한 번에 계산하는 공식이다.

```python
import numpy as np
from sklearn.linear_model import LinearRegression

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
# 약간의 노이즈가 섞인 가짜 선형 데이터 생성

lin_reg = LinearRegression()
lin_reg.fit(X, y)

lin_reg.intercept_, lin_reg.coef_
# intercept_: θ₀ (직선이 y축과 만나는 지점)
# coef_: θ₁ (직선의 기울기)
```

## 2. 경사 하강법
손실 함수가 줄어드는 방향으로 파라미터를 조금씩 이동시켜 최적점을 찾는 방법. 안개 낀 산에서 가장 가파른 내리막으로 한 걸음씩 내려가는 것과 같다. **학습률**이 너무 크면 발산하고, 너무 작으면 학습이 느리다.

```python
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.01)
# eta0: 학습률
# max_iter: 데이터 전체를 최대 몇 번 훑을지
# penalty=None: 규제 없이 순수 경사하강법만 적용

sgd_reg.fit(X, y.ravel())
sgd_reg.intercept_, sgd_reg.coef_
```

세 가지 방식이 있다:
- **배치 경사하강법**: 매 스텝마다 전체 데이터 사용 (정확하지만 데이터 많으면 느림)
- **확률적 경사하강법(SGD)**: 매 스텝마다 랜덤으로 하나만 사용 (빠르지만 불안정)
- **미니배치 경사하강법**: 매 스텝마다 일부만 사용 (실용적인 절충안)

## 3. 다항 회귀
직선이 아니라 곡선을 맞추고 싶을 때, x², x³ 같은 항을 새 특성으로 추가한 뒤 선형 회귀처럼 푼다.

```python
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
# 원래 x 옆에 x^2를 새 특성으로 추가

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
```

⚠️ 주의: 차수를 너무 높이면 데이터에 과하게 맞춰져서(과대적합) 곡선이 노이즈까지 따라가며 새 데이터에는 안 맞게 된다.

## 4. 학습 곡선
훈련 데이터 양을 늘려가며 오차 변화를 그려서 **과소적합**(훈련·검증 오차 둘 다 높음)과 **과대적합**(훈련 오차는 낮은데 검증 오차는 높음)을 진단한다.

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, valid_scores = learning_curve(
    LinearRegression(), X, y, train_sizes=np.linspace(0.01, 1.0, 40), cv=5,
    scoring="neg_root_mean_squared_error")
```

이는 **편향-분산 트레이드오프**와 연결된다: 모델이 너무 단순하면 편향(bias)이 크고, 너무 복잡하면 분산(variance)이 크다. 둘 사이 균형점을 찾는 게 목표다.

## 5. 규제가 있는 선형 모델
가중치가 너무 커지지 않게 눌러서 과대적합을 방지한다.

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

ridge_reg = Ridge(alpha=1, solver="cholesky")
# 릿지(L2): 가중치 제곱합을 페널티로 추가 → 가중치를 0에 가깝게 축소

lasso_reg = Lasso(alpha=0.1)
# 라쏘(L1): 가중치 절댓값 합을 페널티로 추가 → 일부 가중치를 정확히 0으로 만듦 (특성 선택 효과)

elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
# 엘라스틱넷: 릿지와 라쏘를 l1_ratio 비율로 혼합
```

**조기 종료**도 규제 기법 중 하나다: 검증 오차가 다시 오르기 시작하는 순간 훈련을 멈춰서 과대적합을 막는다.

## 6. 로지스틱 회귀
이름은 "회귀"지만 실제로는 **분류**에 쓰인다. 선형 회귀 결과를 **시그모이드 함수**에 통과시켜 0~1 사이 확률로 바꾼다.

```python
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train_5)
log_reg.predict_proba([X_train[0]])
# [0.2, 0.8]처럼 출력 → 80% 확률로 5임
```

**소프트맥스 회귀**는 이를 다중 클래스로 확장한 버전으로, 여러 클래스에 대한 확률을 동시에 계산하며 전부 합하면 1이 된다.

```python
softmax_reg = LogisticRegression(multi_class="multinomial")
softmax_reg.fit(X_train, y_train)
# 예/아니오가 아니라 0~9 중 하나를 직접 예측
```

## 퀴즈 / 몰랐던 부분
**Q.** 로지스틱 회귀는 분류에 쓰이는데 왜 이름이 "회귀"일까?

**A.** 내부적으로는 선형 회귀처럼 입력값의 가중합을 계산해 연속적인 숫자를 먼저 만들고, 마지막 단계에서만 시그모이드 함수와 확률 임계값을 이용해 그 숫자를 클래스로 변환하기 때문이다. 수학적 메커니즘은 회귀이고, 최종 활용 목적만 분류다.

---

# 5장. 서포트 벡터 머신 (SVM)

## 1. 선형 SVM 분류
SVM은 두 클래스를 나누는 경계선 중, 가장 가까운 데이터(**서포트 벡터**)까지의 여백(**마진**)이 최대가 되는 경계를 찾는다.

```python
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

svm_clf = make_pipeline(StandardScaler(), LinearSVC(C=1, random_state=42))
# StandardScaler: SVM은 특성 스케일에 민감하므로 먼저 정규화
# C: 마진의 "부드러움"을 조절 → C가 작을수록 마진은 넓어지지만 오류를 더 허용

svm_clf.fit(X, y)
```

- **하드 마진**: 오류 없이 완벽하게 나눔 (이상치에 매우 민감)
- **소프트 마진**: 약간의 오분류를 허용하는 대신 더 넓고 일반화 잘 되는 마진 확보 (`C`로 조절)

## 2. 비선형 SVM 분류
직선으로 나눌 수 없는 데이터를, **커널 트릭**을 이용해 실제로 고차원으로 변환하는 계산 없이도 그런 효과를 얻는다.

```python
from sklearn.svm import SVC

poly_kernel_svm_clf = make_pipeline(
    StandardScaler(),
    SVC(kernel="poly", degree=3, coef0=1, C=5)
)
# 다항식 커널: 다항 특성(x^2, x^3...)을 추가한 것과 같은 효과를 명시적 계산 없이 흉내냄
poly_kernel_svm_clf.fit(X, y)

rbf_kernel_svm_clf = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", gamma=5, C=0.001)
)
# RBF(가우시안) 커널: 거리 기반 유사도를 측정 → 매우 유연하고 구불구불한 경계 가능
# gamma: 경계가 얼마나 구불구불해질지 조절 (값이 클수록 과대적합 위험 증가)
rbf_kernel_svm_clf.fit(X, y)
```

## 3. SVM 회귀
분류와 반대로, 마진 "안에" 최대한 많은 데이터가 들어오도록 하는 회귀 방식이다.

```python
from sklearn.svm import LinearSVR

svm_reg = make_pipeline(StandardScaler(), LinearSVR(epsilon=0.5, random_state=42))
# epsilon: 마진(튜브)의 폭을 정의 → 이 안에 들어온 점은 오류로 취급하지 않음

svm_reg.fit(X, y)
```

## 4. SVM 이론
"마진을 최대화한다"는 것은 수학적으로 **가중치 벡터의 크기(‖w‖)를 최소화**하는 것과 같다. ‖w‖가 작을수록 마진은 넓어진다. 따라서 최적화 문제는 "모든 데이터를 올바르게 분류하면서(또는 소프트 마진의 경우 약간의 여유를 허용하면서) ‖w‖를 최소화"하는 것이 된다.

## 5. 쌍대 문제
원래 문제(**원문제**, primal)가 풀기 어려울 때, 수학적으로 동일한 답을 주는 다른 형태의 문제인 **쌍대 문제**로 바꿔서 푸는 기법이다.

SVM에서 쌍대 문제가 중요한 이유:
- 쌍대 문제는 데이터 포인트 간의 내적(dot product)만 필요하므로, **커널 트릭**을 자연스럽게 적용할 수 있다.
- 훈련 샘플 수가 특성 수보다 적을 때 원문제보다 더 빠르게 풀린다.

비유: "직접 길을 찾기 어려우면, 수학적으로 똑같은 답이 나오는 다른 지도를 보고 더 효율적으로 찾는 것"과 같다.

## 퀴즈 / 몰랐던 부분
**Q.** RBF 커널 SVM에서 `gamma` 값을 높이면 왜 과대적합 위험이 커질까?

**A.** `gamma`가 커지면 각 훈련 데이터가 영향을 미치는 범위가 좁아져서, 결정 경계가 전체적인 패턴을 반영하는 대신 개별 데이터 포인트 하나하나에 딱 맞게 구불구불해지기 때문이다. 즉 일반화 대신 훈련 데이터를 "암기"하는 상태가 된다.

---

## 참고
- 공식 예제 코드: https://github.com/ageron/handson-ml3
