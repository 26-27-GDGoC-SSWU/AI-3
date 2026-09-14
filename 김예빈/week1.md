# 3장. 분류 (Classification)

## 1. MNIST

### 이론

**MNIST**는 손으로 쓴 숫자 이미지로 구성된 대표적인 머신러닝
데이터셋이다.

-   총 70,000개의 작은 숫자 이미지로 구성되어 있다.
-   각 이미지는 `28 × 28` 픽셀이다.
-   따라서 하나의 이미지는 총 784개의 특성을 가진다.
-   머신러닝 및 분류 알고리즘의 학습과 테스트에 많이 사용된다.
-   사이킷런의 `fetch_openml()` 함수를 이용해 내려받을 수 있다.
-   노트북에서는 앞의 60,000개를 훈련 세트, 나머지 10,000개를
    테스트 세트로 사용한다.

### 데이터 불러오기

``` python
from sklearn.datasets import fetch_openml

mnist = fetch_openml("mnist_784", as_frame=False)

X, y = mnist.data, mnist.target
```

데이터의 크기를 확인하면 다음과 같다.

``` python
X.shape
# (70000, 784)

y.shape
# (70000,)
```

`X`에는 이미지의 픽셀 정보가 들어 있고, `y`에는 각 이미지에 해당하는
숫자 레이블이 들어 있다.

### 숫자 이미지 확인하기

``` python
import matplotlib.pyplot as plt

def plot_digit(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")

some_digit = X[0]
plot_digit(some_digit)
plt.show()
```

784개의 값으로 펼쳐진 데이터를 `28 × 28` 배열로 다시 변환하면 실제 숫자
이미지로 확인할 수 있다.

### 훈련 세트와 테스트 세트 분리

``` python
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]
```

------------------------------------------------------------------------

## 2. 이진 분류기 훈련

### 이론

**이진 분류(Binary Classification)**는 데이터를 두 개의 클래스로
구분하는 문제이다.

예를 들어 MNIST에서 다음과 같은 문제를 만들 수 있다.

-   `5`인 이미지 → **양성(True)**
-   `5`가 아닌 이미지 → **음성(False)**

즉, `"5인가?"`라는 하나의 질문에 대해 두 가지 결과만 출력하는
분류기이다.

### 5와 5가 아닌 숫자 구분하기

``` python
y_train_5 = (y_train == "5")
y_test_5 = (y_test == "5")
```

### SGDClassifier

노트북에서는 **확률적 경사 하강법(Stochastic Gradient Descent)**을
사용하는 `SGDClassifier`를 이용한다.

`SGDClassifier`의 특징은 다음과 같다.

-   큰 데이터셋을 효율적으로 처리할 수 있다.
-   훈련 샘플을 독립적으로 처리하기 때문에 온라인 학습에도 적합하다.

``` python
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
```

예측은 다음과 같이 수행한다.

``` python
sgd_clf.predict([some_digit])
```

------------------------------------------------------------------------

## 3. 성능 측정

분류 모델은 단순히 정확도 하나만으로 평가하기 어려운 경우가 많다.\
특히 클래스 비율이 불균형한 데이터에서는 **오차 행렬, 정밀도, 재현율, F1
점수, ROC 곡선** 등을 함께 살펴봐야 한다.

------------------------------------------------------------------------

## 3.1 교차 검증을 사용한 정확도 측정

### 이론

교차 검증은 훈련 데이터를 여러 개의 폴드로 나눈 뒤, 일부를 학습에
사용하고 나머지를 검증에 사용하여 모델의 성능을 평가하는 방법이다.

사이킷런에서는 `cross_val_score()`를 사용할 수 있다.

``` python
from sklearn.model_selection import cross_val_score

cross_val_score(
    sgd_clf,
    X_train,
    y_train_5,
    cv=3,
    scoring="accuracy"
)
```

여기서 `cv=3`은 데이터를 3개의 폴드로 나누어 교차 검증한다는 의미이다.

### 정확도만 보면 안 되는 이유

노트북에서는 `DummyClassifier`를 이용해 항상 가장 빈번한 클래스를
예측하는 단순한 모델도 확인한다.

``` python
from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier()
dummy_clf.fit(X_train, y_train_5)

cross_val_score(
    dummy_clf,
    X_train,
    y_train_5,
    cv=3,
    scoring="accuracy"
)
```

MNIST에서 숫자 5는 전체 데이터 중 일부이므로, 무조건 `"5가 아니다"`라고
예측해도 높은 정확도가 나올 수 있다.

따라서 **불균형 데이터에서는 정확도만으로 모델을 평가하면 성능을 잘못
해석할 수 있다.**

------------------------------------------------------------------------

## 3.2 오차 행렬

### 이론

**오차 행렬(Confusion Matrix)**은 분류기가 어떤 클래스를 어떤 클래스로
잘못 분류했는지 확인하는 방법이다.

모든 A/B 쌍에 대해 **실제 클래스 A의 샘플이 클래스 B로 분류된 횟수**를
센다.

오차 행렬을 만들려면 실제 타깃과 비교할 예측값이 필요하다.\
교차 검증을 이용한 예측값은 `cross_val_predict()`로 만들 수 있다.

``` python
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(
    sgd_clf,
    X_train,
    y_train_5,
    cv=3
)
```

이후 `confusion_matrix()`를 사용한다.

``` python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_train_5, y_train_pred)
cm
```

### 오차 행렬 구조

오차 행렬에서는

-   **행(row)**: 실제 클래스
-   **열(column)**: 예측 클래스

를 나타낸다.

이진 분류의 오차 행렬은 다음 네 가지 값으로 해석할 수 있다.

  구분                  의미
  --------------------- ------------------------------------
  TN (True Negative)    실제 음성을 음성으로 정확하게 예측
  FP (False Positive)   실제 음성을 양성으로 잘못 예측
  FN (False Negative)   실제 양성을 음성으로 잘못 예측
  TP (True Positive)    실제 양성을 양성으로 정확하게 예측

------------------------------------------------------------------------

## 3.3 정밀도와 재현율

### 정밀도 (Precision)

**정밀도**는 모델이 **양성이라고 예측한 샘플 중 실제로 양성인 샘플의
비율**이다.

``` text
Precision = TP / (TP + FP)
```

사이킷런에서는 `precision_score()`를 사용한다.

``` python
from sklearn.metrics import precision_score

precision_score(y_train_5, y_train_pred)
```

즉, `"5라고 예측한 것들이 얼마나 정확한가?"`를 나타낸다.

### 재현율 (Recall)

**재현율**은 **실제 양성 샘플 중 분류기가 정확하게 찾아낸 양성 샘플의
비율**이다.

``` text
Recall = TP / (TP + FN)
```

``` python
from sklearn.metrics import recall_score

recall_score(y_train_5, y_train_pred)
```

즉, `"실제 5 중에서 얼마나 많은 5를 찾아냈는가?"`를 나타낸다.

### F1 점수

정밀도와 재현율을 하나의 값으로 비교하고 싶을 때 **F1 점수**를 사용할 수
있다.

F1 점수는 정밀도와 재현율의 **조화 평균**이다.

``` python
from sklearn.metrics import f1_score

f1_score(y_train_5, y_train_pred)
```

정밀도와 재현율이 모두 높을 때 F1 점수도 높아진다.

------------------------------------------------------------------------

## 3.4 정밀도/재현율 트레이드오프

### 이론

분류기는 내부적으로 계산한 **결정 점수(decision score)**를 임곗값과
비교하여 클래스를 결정한다.

``` python
y_scores = sgd_clf.decision_function([some_digit])
```

임곗값을 직접 적용할 수도 있다.

``` python
threshold = 0
y_some_digit_pred = (y_scores > threshold)
```

임곗값을 높이면 양성으로 판단하기 어려워지기 때문에 일반적으로

-   **정밀도는 증가**
-   **재현율은 감소**

한다.

반대로 임곗값을 낮추면

-   더 많은 샘플을 양성으로 판단하여 **재현율은 증가**
-   거짓 양성도 늘어날 수 있어 **정밀도는 감소**

한다. 이를 **정밀도/재현율 트레이드오프**라고 한다.

### 여러 임곗값의 점수 구하기

``` python
y_scores = cross_val_predict(
    sgd_clf,
    X_train,
    y_train_5,
    cv=3,
    method="decision_function"
)
```

``` python
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(
    y_train_5,
    y_scores
)
```

`precision_recall_curve()`를 이용하면 여러 임곗값에서 정밀도와 재현율이
어떻게 변하는지 확인할 수 있다.

### 원하는 정밀도에 맞춰 임곗값 선택하기

노트북에서는 정밀도 90% 이상을 만족하는 임곗값을 찾는다.

``` python
idx_for_90_precision = (precisions >= 0.90).argmax()
threshold_for_90_precision = thresholds[idx_for_90_precision]

y_train_pred_90 = (y_scores >= threshold_for_90_precision)
```

이처럼 모델의 목적에 따라 적절한 임곗값을 선택할 수 있다.

------------------------------------------------------------------------

## 3.5 ROC 곡선

### 이론

**ROC(Receiver Operating Characteristic) 곡선**은 이진 분류기의 성능을
확인할 때 널리 사용하는 도구이다.

ROC 곡선은 다음 두 값을 이용한다.

-   **TPR(True Positive Rate)**: 진짜 양성 비율 = 재현율
-   **FPR(False Positive Rate)**: 실제 음성 중 양성으로 잘못 분류된 비율

FPR은 `1 - TNR`과 같다.

-   **TNR(True Negative Rate)**: 진짜 음성 비율
-   TNR은 **특이도(Specificity)**라고도 한다.

따라서 ROC 곡선은 **민감도(재현율)에 대한 1-특이도 그래프**라고 볼 수
있다.

### ROC 곡선 계산

``` python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
```

이후 Matplotlib을 이용해 FPR에 대한 TPR을 그릴 수 있다.

### ROC AUC

ROC 곡선 아래의 면적을 **AUC(Area Under the Curve)**라고 한다.

``` python
from sklearn.metrics import roc_auc_score

roc_auc_score(y_train_5, y_scores)
```

좋은 분류기일수록 ROC 곡선이 왼쪽 위에 가까워지고 AUC 값도 커진다.

### RandomForest와 비교

노트북에서는 `RandomForestClassifier`의 예측 확률도 교차 검증으로 구해
SGD 분류기와 비교한다.

``` python
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)

y_probas_forest = cross_val_predict(
    forest_clf,
    X_train,
    y_train_5,
    cv=3,
    method="predict_proba"
)

y_scores_forest = y_probas_forest[:, 1]
```

모델을 비교할 때 ROC-AUC뿐 아니라 **Precision-Recall 곡선**도 함께
살펴보는 것이 유용하다.

------------------------------------------------------------------------

## 4. 다중 분류

### 이론

**다중 분류(Multiclass Classification)**는 둘 이상의 클래스를 구별하는
문제이다.

MNIST에서는 숫자 `0~9` 중 하나를 예측해야 하므로 대표적인 다중 분류
문제이다.

일부 알고리즘은 다중 클래스를 직접 처리할 수 있다.

-   `LogisticRegression`
-   `RandomForestClassifier`
-   `SGDClassifier` 등

반면 이진 분류 알고리즘을 여러 개 조합하여 다중 분류를 수행할 수도 있다.

### OvR (One-versus-the-Rest)

각 클래스마다 `"해당 클래스인가 / 아닌가"`를 구분하는 이진 분류기를
하나씩 만든다.

MNIST라면 다음과 같은 분류기를 만들 수 있다.

``` text
0 vs 나머지
1 vs 나머지
2 vs 나머지
...
9 vs 나머지
```

예측할 때 각 분류기의 결정 점수를 확인하고 **가장 높은 점수를 가진
클래스를 선택**한다. 대부분의 이진 분류 알고리즘에서는 OvR 전략을 많이 사용한다.

### OvO (One-versus-One)

모든 클래스의 **두 클래스 조합마다** 이진 분류기를 훈련한다.

MNIST처럼 클래스가 10개라면 총 45개의 이진 분류기가 필요하다.

각 분류기는 전체 데이터가 아니라 **구별해야 하는 두 클래스의 샘플만
사용하면 된다.**  따라서 작은 훈련 세트에서 여러 분류기를 훈련시키는 것이 유리한
알고리즘에서 사용할 수 있다.

### SVC를 이용한 다중 분류

노트북에서는 SVC를 사용한다.

``` python
from sklearn.svm import SVC

svm_clf = SVC(random_state=42)
svm_clf.fit(X_train[:2000], y_train[:2000])

svm_clf.predict([some_digit])
```

SVC는 내부적으로 **OvO 전략으로 훈련**한다.

결정 점수도 확인할 수 있다.

``` python
some_digit_scores = svm_clf.decision_function([some_digit])
some_digit_scores
```

가장 높은 점수의 클래스를 선택할 수 있다.

``` python
class_id = some_digit_scores.argmax()
svm_clf.classes_[class_id]
```

### OvR 전략을 명시적으로 사용하기

``` python
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

ovr_clf = OneVsRestClassifier(SVC(random_state=42))
ovr_clf.fit(X_train[:2000], y_train[:2000])

ovr_clf.predict([some_digit])
```

### SGDClassifier를 이용한 다중 분류

``` python
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)

sgd_clf.predict([some_digit])
```

각 클래스에 대한 결정 점수도 확인할 수 있다.

``` python
sgd_clf.decision_function([some_digit])
```

### 스케일링

노트북에서는 입력 특성을 표준화한 후 SGD 분류기의 성능을 다시 평가한다.

``` python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype("float64"))

cross_val_score(
    sgd_clf,
    X_train_scaled,
    y_train,
    cv=3,
    scoring="accuracy"
)
```

특성 스케일링을 통해 모델의 성능이 개선될 수 있다.

------------------------------------------------------------------------

## 5. 에러 분석

모델의 성능을 개선하려면 단순히 하나의 점수만 보는 것이 아니라 **어떤
클래스에서 오류가 발생하는지 분석**해야 한다.

### 다중 클래스 오차 행렬

``` python
from sklearn.metrics import ConfusionMatrixDisplay

y_train_pred = cross_val_predict(
    sgd_clf,
    X_train_scaled,
    y_train,
    cv=3
)

ConfusionMatrixDisplay.from_predictions(
    y_train,
    y_train_pred
)

plt.show()
```

### 정규화된 오차 행렬

클래스마다 샘플 개수가 다르면 원래 개수 대신 비율을 보는 것이 편리하다.

``` python
ConfusionMatrixDisplay.from_predictions(
    y_train,
    y_train_pred,
    normalize="true",
    values_format=".0%"
)

plt.show()
```

오차 행렬을 분석하면 특정 숫자끼리 자주 혼동되는지 확인할 수 있고, 이를
바탕으로 데이터 전처리나 모델 개선 방향을 찾을 수 있다.

------------------------------------------------------------------------

## 6. 다중 레이블 분류

### 이론

**다중 레이블 분류(Multilabel Classification)**는 하나의 샘플에 대해
**여러 개의 이진 꼬리표(label)를 동시에 출력**하는 분류 문제이다.

예를 들어 하나의 숫자 이미지에 대해 다음 두 질문을 동시에 할 수 있다.

1.  숫자가 `7 이상`인가?
2.  숫자가 `홀수`인가?

하나의 이미지가 두 조건을 동시에 만족할 수도 있으므로 결과는 하나의
클래스가 아니라 여러 레이블로 표현된다.

### 다중 레이블 생성

``` python
import numpy as np

y_train_large = (y_train >= "7")
y_train_odd = (y_train.astype("int8") % 2 == 1)

y_multilabel = np.c_[y_train_large, y_train_odd]
```

### KNeighborsClassifier 사용

``` python
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)

knn_clf.predict([some_digit])
```

예측 결과는 각 레이블에 대한 True/False 값으로 출력된다.

### 다중 레이블 성능 평가

노트북에서는 각 레이블의 F1 점수를 계산한 뒤 평균을 내는 `macro` 방식을
사용한다.

``` python
y_train_knn_pred = cross_val_predict(
    knn_clf,
    X_train,
    y_multilabel,
    cv=3
)

f1_score(
    y_multilabel,
    y_train_knn_pred,
    average="macro"
)
```

`average="weighted"`를 사용하면 각 레이블의 샘플 수를 고려해 가중 평균을
계산할 수도 있다.

### ClassifierChain

노트북에서는 한 분류기의 예측 결과를 다음 분류기의 입력으로 활용하는
`ClassifierChain`도 사용한다.

``` python
from sklearn.multioutput import ClassifierChain
from sklearn.svm import SVC

chain_clf = ClassifierChain(SVC(), cv=3, random_state=42)
chain_clf.fit(X_train[:2000], y_multilabel[:2000])

chain_clf.predict([some_digit])
```

레이블 사이에 연관성이 있을 때 앞선 레이블의 예측을 다음 레이블 예측에
활용할 수 있다.

------------------------------------------------------------------------

## 7. 다중 출력 분류

### 이론

**다중 출력 분류(Multioutput Classification)**는 다중 레이블 분류를
일반화한 형태이다.

다중 레이블에서는 각 레이블이 보통 이진 값이지만, 다중 출력에서는 **각
출력이 여러 개의 값을 가질 수 있다.**

노트북에서는 숫자 이미지에 노이즈를 추가한 뒤, 원래의 깨끗한 이미지를
예측하는 문제를 만든다.

### 노이즈 데이터 생성

``` python
np.random.seed(42)

noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise

noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise

y_train_mod = X_train
y_test_mod = X_test
```

입력은 **노이즈가 포함된 숫자 이미지**, 타깃은 **원래의 깨끗한 숫자
이미지**가 된다.

각 픽셀이 하나의 출력 레이블 역할을 하며, 각 픽셀은 여러 픽셀 값을 가질
수 있다.

### KNN으로 이미지 복원

``` python
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_mod, y_train_mod)

clean_digit = knn_clf.predict([X_test_mod[0]])
plot_digit(clean_digit)
plt.show()
```

이를 통해 분류가 단순히 `"하나의 입력 → 하나의 클래스"`만을 의미하는
것이 아니라 여러 개의 출력값을 예측하는 문제로 확장될 수 있음을 확인할
수 있다.

------------------------------------------------------------------------

# 4장. 모델 훈련 (Training Models)

------------------------------------------------------------------------

## 1. 선형 회귀

### 이론

**선형 회귀(Linear Regression)**는 입력 특성의 가중치 합에
**편향(bias)**이라는 상수를 더해 예측을 만드는 모델이다.

``` text
예측값 = 편향 + (특성1 × 가중치1) + (특성2 × 가중치2) + ...
```

즉, 모델을 훈련한다는 것은 데이터에 가장 잘 맞는 **모델
파라미터(가중치와 편향)**를 찾는 과정이다.

회귀에서 널리 사용하는 성능 측정 지표 중 하나는 **평균 제곱근
오차(RMSE)**이다.

``` text
RMSE = 예측값과 실제값의 차이를 제곱한 뒤 평균을 내고 제곱근을 취한 값
```

따라서 선형 회귀 모델을 훈련할 때는 훈련 데이터에 대한 오차가 작아지도록
적절한 파라미터를 찾아야 한다.

### 실습 데이터 생성

노트북에서는 선형 관계를 가지는 임의의 데이터를 생성하여 선형 회귀를
실습한다.

``` python
import numpy as np

np.random.seed(42)
m = 100

X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1)
```

이 데이터는 대략 다음 관계를 가진다.

``` text
y ≈ 4 + 3x + noise
```

------------------------------------------------------------------------

## 2. 정규 방정식

### 이론

**정규 방정식(Normal Equation)**은 선형 회귀의 비용 함수를 최소화하는
모델 파라미터를 직접 계산하기 위한 수학적 방법이다.

경사 하강법처럼 반복적으로 파라미터를 수정하지 않고 계산을 통해 해를
구할 수 있다는 특징이 있다.

노트북에서는 편향을 포함하기 위해 입력 데이터에 `1`을 추가한 뒤
파라미터를 계산한다.

``` python
from sklearn.preprocessing import add_dummy_feature

X_b = add_dummy_feature(X)
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
```

예측은 다음과 같이 할 수 있다.

``` python
X_new = np.array([[0], [2]])
X_new_b = add_dummy_feature(X_new)

y_predict = X_new_b @ theta_best
```

### 사이킷런의 LinearRegression

실제로는 직접 정규 방정식을 구현하기보다 `LinearRegression`을 사용할 수
있다.

``` python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)

lin_reg.intercept_, lin_reg.coef_
```

예측은 다음과 같이 수행한다.

``` python
lin_reg.predict(X_new)
```

노트북의 `LinearRegression`은 내부적으로 최소제곱 문제를 풀며,
`scipy.linalg.lstsq()`를 사용한다.

또한 유사역행렬(pseudoinverse)을 이용해서도 같은 해를 계산할 수 있다.

``` python
theta_best_svd = np.linalg.pinv(X_b) @ y
```

정규 방정식 계열의 방법은 모델을 한 번 학습한 뒤에는 **예측이 매우
빠르다.**

------------------------------------------------------------------------

## 3. 경사 하강법

### 이론

**경사 하강법(Gradient Descent)**은 여러 종류의 문제에서 최적의 해를
찾는 데 사용할 수 있는 일반적인 최적화 알고리즘이다.

핵심 아이디어는 다음과 같다.

``` text
현재 파라미터 설정
      ↓
비용 함수의 기울기 계산
      ↓
비용이 감소하는 방향으로 파라미터 이동
      ↓
반복
      ↓
비용 함수의 최솟값에 접근
```

즉, 비용 함수를 최소화하기 위해 모델 파라미터를 반복해서 조정한다.

### 학습률

**학습률(Learning Rate)**은 한 번의 경사 하강법 스텝에서 얼마나 크게
이동할지를 결정하는 하이퍼파라미터이다.

-   학습률이 **너무 작으면**
    -   수렴하기 위해 많은 반복이 필요하다.
    -   학습 시간이 오래 걸린다.
-   학습률이 **너무 크면**
    -   최솟값을 지나쳐 버릴 수 있다.
    -   알고리즘이 발산할 수 있다.

따라서 적절한 학습률을 선택하는 것이 중요하다.

------------------------------------------------------------------------

## 3.1 배치 경사 하강법

### 이론

**편도함수(Partial Derivative)**는 하나의 파라미터가 조금 변했을 때 비용
함수가 얼마나 변하는지 계산한다.

모든 파라미터에 대한 편도함수를 모으면 **그래디언트(gradient)**를 얻을
수 있다.

**배치 경사 하강법(Batch Gradient Descent)**은 매 경사 하강법 스텝에서
**전체 훈련 세트**를 사용하여 그래디언트를 계산한다.

### 특징

-   매 스텝에서 전체 훈련 데이터를 사용한다.
-   안정적으로 최솟값을 향해 이동한다.
-   훈련 세트가 매우 크면 한 스텝의 계산 비용이 커져 느릴 수 있다.

### 구현

``` python
eta = 0.1
n_epochs = 1000
m = len(X_b)

np.random.seed(42)
theta = np.random.randn(2, 1)

for epoch in range(n_epochs):
    gradients = 2 / m * X_b.T @ (X_b @ theta - y)
    theta = theta - eta * gradients
```

학습이 끝난 `theta`는 정규 방정식으로 구한 값과 매우 비슷한 값에
수렴한다.

------------------------------------------------------------------------

## 3.2 확률적 경사 하강법

### 이론

**확률적 경사 하강법(Stochastic Gradient Descent, SGD)**은 매 스텝에서
훈련 샘플 **한 개를 무작위로 선택**하여 그 샘플에 대한 그래디언트를
계산한다.

### 장점

-   한 번에 하나의 샘플만 처리한다.
-   매 반복에서 처리해야 하는 데이터가 적어 빠르다.
-   매우 큰 데이터셋에도 적용하기 좋다.
-   무작위성 때문에 지역 최솟값에서 탈출하는 데 도움이 될 수 있다.

### 단점

-   배치 경사 하강법보다 움직임이 불안정하다.
-   최솟값에 도달한 뒤에도 계속 위아래로 움직일 수 있다.

이를 해결하기 위해 학습이 진행될수록 학습률을 점차 감소시키는 **학습
스케줄(learning schedule)**을 사용할 수 있다.

### SGDRegressor

사이킷런에서는 `SGDRegressor`를 이용할 수 있다.

``` python
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(
    max_iter=1000,
    tol=1e-5,
    penalty=None,
    eta0=0.01,
    n_iter_no_change=100,
    random_state=42
)

sgd_reg.fit(X, y.ravel())
```

훈련된 모델의 파라미터는 다음과 같이 확인할 수 있다.

``` python
sgd_reg.intercept_, sgd_reg.coef_
```

------------------------------------------------------------------------

## 3.3 미니배치 경사 하강법

### 이론

**미니배치 경사 하강법(Mini-batch Gradient Descent)**은 전체 훈련
세트도, 하나의 샘플도 아닌 **작은 샘플 묶음(mini-batch)**을 사용하여
그래디언트를 계산한다.

``` text
배치 GD       → 전체 훈련 데이터 사용
SGD           → 샘플 1개 사용
미니배치 GD   → 작은 샘플 묶음 사용
```

### 특징

-   SGD보다 안정적으로 이동한다.
-   배치 경사 하강법보다 한 스텝의 계산량이 적다.
-   행렬 연산에 최적화된 하드웨어와 **GPU**를 효율적으로 활용할 수 있다.

### 선형 회귀 학습 방법 비교

정규 방정식, 배치 경사 하강법, 확률적 경사 하강법, 미니배치 경사
하강법은 **파라미터를 찾는 방법은 다르지만**, 충분히 잘 학습되면 매우
비슷한 선형 회귀 모델을 만든다.

  방법               한 스텝에서 사용하는 데이터   특징
  ------------------ ----------------------------- -----------------------------------
  정규 방정식 계열   전체 데이터                   반복 없이 직접 해 계산
  배치 GD            전체 데이터                   안정적이지만 큰 데이터에서 느림
  SGD                샘플 1개                      빠르지만 불안정
  미니배치 GD        작은 샘플 묶음                GPU 활용에 유리하고 비교적 안정적

------------------------------------------------------------------------

## 4. 다항 회귀

### 이론

데이터가 단순한 직선 형태가 아니라면 **다항 회귀(Polynomial
Regression)**를 사용할 수 있다.

다항 회귀는 데이터의 각 특성의 **거듭제곱을 새로운 특성으로 추가**한 뒤,
확장된 데이터셋에 선형 모델을 훈련시키는 방법이다.

예를 들어 원래 특성이 `x` 하나라면 다음과 같이 확장할 수 있다.

``` text
x → x, x²
```

모델 자체는 선형 회귀이지만 입력 특성을 확장했기 때문에 곡선 형태의
데이터를 학습할 수 있다.

### 실습 데이터 생성

노트북에서는 2차 방정식 형태의 데이터를 만든다.

``` python
np.random.seed(42)

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)
```

### PolynomialFeatures

``` python
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(
    degree=2,
    include_bias=False
)

X_poly = poly_features.fit_transform(X)
```

`degree=2`로 설정하면 원래 특성 `x`에 `x²` 특성이 추가된다.

이후 일반 선형 회귀 모델을 학습한다.

``` python
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
```

### 너무 높은 차수의 문제

다항 차수를 지나치게 높이면 훈련 데이터에 지나치게 맞춰지는
**과대적합(overfitting)**이 발생할 수 있다.

반대로 모델이 너무 단순하면 데이터의 패턴을 충분히 학습하지 못하는
**과소적합(underfitting)**이 발생할 수 있다.

------------------------------------------------------------------------

## 5. 학습 곡선

노트북에서는 모델의 과대적합과 과소적합을 분석하기 위해 **학습
곡선(Learning Curve)**을 사용한다.

학습 곡선은 훈련 세트의 크기를 늘려가면서

-   훈련 오차
-   검증 오차

가 어떻게 변하는지 확인한다.

``` python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, valid_scores = learning_curve(
    LinearRegression(),
    X,
    y,
    train_sizes=np.linspace(0.01, 1.0, 40),
    cv=5,
    scoring="neg_root_mean_squared_error"
)

train_errors = -train_scores.mean(axis=1)
valid_errors = -valid_scores.mean(axis=1)
```

### 학습 곡선 해석

**과소적합**된 모델은 일반적으로

-   훈련 오차가 높고
-   검증 오차도 높으며
-   두 오차가 서로 비슷해진다.

**과대적합**된 모델은 일반적으로

-   훈련 오차가 매우 낮지만
-   검증 오차는 상대적으로 높아
-   두 곡선 사이에 차이가 발생한다.

노트북에서는 10차 다항 회귀의 학습 곡선도 확인한다.

``` python
from sklearn.pipeline import make_pipeline

polynomial_regression = make_pipeline(
    PolynomialFeatures(degree=10, include_bias=False),
    LinearRegression()
)
```

------------------------------------------------------------------------

## 6. 규제가 있는 선형 모델

모델의 과대적합을 줄이는 대표적인 방법은 **규제(Regularization)**를
적용하는 것이다.

규제는 모델이 훈련 데이터에 지나치게 복잡하게 맞춰지는 것을 막기 위해
모델의 가중치를 제한한다.

대표적인 규제 선형 모델은 다음과 같다.

-   릿지 회귀
-   라쏘 회귀
-   엘라스틱넷

------------------------------------------------------------------------

## 6.1 릿지 회귀

### 이론

**릿지 회귀(Ridge Regression)**는 규제가 추가된 선형 회귀 버전이다.

훈련 비용 함수에 **L2 규제항**을 추가하여 모델의 가중치가 가능한 한 작게
유지되도록 한다.

규제항은 **훈련하는 동안에만 비용 함수에 추가**한다.

훈련이 끝난 뒤 모델의 실제 성능을 평가할 때는 규제가 포함되지 않은 성능
지표를 사용한다.

### alpha

릿지 회귀의 `alpha`는 모델에 얼마나 강한 규제를 적용할지 조절한다.

-   `alpha = 0` → 일반 선형 회귀와 동일
-   `alpha`가 커질수록 → 가중치를 더 강하게 제한

### Ridge 사용

``` python
from sklearn.linear_model import Ridge

ridge_reg = Ridge(
    alpha=0.1,
    solver="cholesky"
)

ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])
```

SGD 방식으로 L2 규제를 적용할 수도 있다.

``` python
sgd_reg = SGDRegressor(
    penalty="l2",
    alpha=0.1 / m,
    tol=None,
    max_iter=1000,
    eta0=0.01,
    random_state=42
)

sgd_reg.fit(X, y.ravel())
```

------------------------------------------------------------------------

## 6.2 라쏘 회귀

### 이론

**라쏘 회귀(Lasso Regression)** 역시 비용 함수에 규제항을 추가하지만,
릿지의 L2 노름 대신 **가중치 벡터의 L1 노름**을 사용한다.

라쏘 회귀의 중요한 특징은 **덜 중요한 특성의 가중치를 0으로 만들 수
있다는 것**이다.

따라서

-   자동으로 특성 선택을 수행할 수 있고
-   중요하지 않은 특성이 제거된 **희소 모델(sparse model)**을 만들 수
    있다.

### Lasso 사용

``` python
from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)

lasso_reg.predict([[1.5]])
```

------------------------------------------------------------------------

## 6.3 엘라스틱넷

### 이론

**엘라스틱넷(Elastic Net)**은 릿지 회귀와 라쏘 회귀를 절충한 모델이다.

규제항은

-   릿지의 L2 규제
-   라쏘의 L1 규제

를 함께 사용한다.

두 규제의 비율은 **혼합 비율(mix ratio)**을 이용해 조절한다.

사이킷런에서는 `l1_ratio`로 설정한다.

``` python
from sklearn.linear_model import ElasticNet

elastic_net = ElasticNet(
    alpha=0.1,
    l1_ratio=0.5
)

elastic_net.fit(X, y)
elastic_net.predict([[1.5]])
```

### 선형 회귀 vs 릿지 vs 라쏘 vs 엘라스틱넷

일반적으로 규제가 전혀 없는 선형 회귀보다 규제가 있는 모델을 사용하는
것이 좋다.

``` text
기본 선택 → Ridge

사용하는 특성이 일부뿐이라고 생각함
        → Lasso 또는 Elastic Net

특성 수가 훈련 샘플 수보다 많거나
특성 몇 개가 강하게 연관되어 있음
        → Elastic Net
```

라쏘는 특성 수가 훈련 샘플 수보다 많거나 특성 사이의 상관관계가 강한
경우 불규칙하게 동작할 수 있으므로, 이런 경우 엘라스틱넷을 선호할 수
있다.

------------------------------------------------------------------------

## 6.4 조기 종료

### 이론

**조기 종료(Early Stopping)**는 검증 오차가 최소가 되는 시점에서 훈련을
중단하여 과대적합을 방지하는 규제 방법이다.

훈련이 진행되면 처음에는

``` text
훈련 진행
   ↓
훈련 오차 감소
검증 오차 감소
```

하지만 지나치게 오래 훈련하면

``` text
훈련 오차는 계속 감소
검증 오차는 다시 증가
        ↓
     과대적합
```

이 발생할 수 있다.

따라서 **검증 오차가 가장 낮았던 모델을 저장**한다.

### 노트북 구현

``` python
from copy import deepcopy
from sklearn.metrics import root_mean_squared_error

sgd_reg = SGDRegressor(
    penalty=None,
    eta0=0.002,
    random_state=42
)

n_epochs = 500
best_valid_rmse = float("inf")

for epoch in range(n_epochs):
    sgd_reg.partial_fit(X_train_prep, y_train)

    y_valid_predict = sgd_reg.predict(X_valid_prep)
    val_error = root_mean_squared_error(
        y_valid,
        y_valid_predict
    )

    if val_error < best_valid_rmse:
        best_valid_rmse = val_error
        best_model = deepcopy(sgd_reg)
```

`partial_fit()`을 사용하면 에포크마다 조금씩 모델을 학습시키면서 검증
오차를 확인할 수 있다.

------------------------------------------------------------------------

## 7. 로지스틱 회귀

### 이론

이름에는 **회귀(Regression)**가 들어가지만, **로지스틱 회귀(Logistic
Regression)**는 주로 분류에 사용된다.

샘플이 특정 클래스에 속할 **확률**을 추정한다.

``` text
추정 확률 ≥ 임곗값
        → 양성 클래스

추정 확률 < 임곗값
        → 음성 클래스
```

선형 회귀처럼 계산된 값을 그대로 출력하지 않고, 계산 결과에 **로지스틱
함수(시그모이드 함수)**를 적용한다.

시그모이드 함수는 어떤 실수값이 입력되어도 **0\~1 사이의 값**을
출력한다.

따라서 이를 확률처럼 해석할 수 있다.

------------------------------------------------------------------------

### 추정 확률

노트북에서는 붓꽃(Iris) 데이터셋을 사용한다.

``` python
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
```

Iris virginica인지 아닌지를 분류하는 이진 분류 문제를 만든다.

``` python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = iris.data[["petal width (cm)"]].values
y = iris.target_names[iris.target] == "virginica"

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=42
)

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
```

확률은 `predict_proba()`로 확인할 수 있다.

``` python
log_reg.predict_proba(X_new)
```

------------------------------------------------------------------------

### 결정 경계

**결정 경계(Decision Boundary)**는 모델이 한 클래스를 다른 클래스로
분류하는 기준이 되는 경계이다.

예측 확률이 임곗값을 넘어가면서 예측 클래스가 변경되는 지점이라고 볼 수
있다.

특성을 두 개 사용하면 결정 경계를 2차원 공간에서 시각화할 수도 있다.

``` python
X = iris.data[
    ["petal length (cm)", "petal width (cm)"]
].values

y = iris.target_names[iris.target] == "virginica"

log_reg = LogisticRegression(C=2, random_state=42)
log_reg.fit(X_train, y_train)
```

------------------------------------------------------------------------

## 8. 소프트맥스 회귀

### 이론

**소프트맥스 회귀(Softmax Regression)**는 로지스틱 회귀를 여러 클래스에
대해 직접 사용할 수 있도록 일반화한 모델이다.

여러 개의 이진 분류기를 별도로 훈련해 연결하는 대신 **하나의 모델이 여러
클래스를 직접 구분**할 수 있다.

### 동작 과정

``` text
입력 샘플
   ↓
각 클래스의 점수 계산
   ↓
Softmax 함수 적용
   ↓
각 클래스의 확률 계산
   ↓
확률이 가장 높은 클래스 선택
```

소프트맥스 함수는 각 클래스의 점수를 확률로 변환하며, 모든 클래스의
확률을 더하면 1이 된다.

### Iris 다중 분류

노트북에서는 꽃잎 길이와 너비를 사용하여 Iris의 세 클래스를 분류한다.

``` python
X = iris.data[
    ["petal length (cm)", "petal width (cm)"]
].values

y = iris["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=42
)

softmax_reg = LogisticRegression(
    C=30,
    random_state=42
)

softmax_reg.fit(X_train, y_train)
```

클래스를 직접 예측할 수 있다.

``` python
softmax_reg.predict([[5, 2]])
```

각 클래스의 확률도 확인할 수 있다.

``` python
softmax_reg.predict_proba([[5, 2]]).round(2)
```

------------------------------------------------------------------------

### 크로스엔트로피 비용 함수

소프트맥스 회귀에서는 **크로스엔트로피(Cross-Entropy)** 비용 함수를
사용한다.

훈련의 목적은

> **실제 타깃 클래스에는 높은 확률을 예측하도록 모델을 학습하는 것**

이다.

크로스엔트로피는 모델이 실제 타깃 클래스에 낮은 확률을 부여하면 큰
비용을 부과한다.

따라서 크로스엔트로피를 최소화하면 모델은 **타깃 클래스에 높은 확률을
예측하는 방향으로 학습**된다.

즉, 크로스엔트로피는 **모델이 추정한 클래스 확률이 실제 타깃과 얼마나 잘
맞는지 측정하는 비용 함수**라고 볼 수 있다.


------------------------------------------------------------------------
# 5장. 서포트 벡터 머신 (Support Vector Machines)

------------------------------------------------------------------------

## 1. 서포트 벡터 머신

### 이론

**서포트 벡터 머신(Support Vector Machine, SVM)**은 다양한 머신러닝
문제에 사용할 수 있는 강력하고 다목적인 모델이다.

SVM은 다음과 같은 작업에 사용할 수 있다.

-   선형 분류
-   비선형 분류
-   회귀
-   특이치 탐지

특히 **중소 규모의 복잡한 데이터셋 분류 문제**에서 좋은 성능을 보인다.

SVM의 핵심 아이디어는 단순히 두 클래스를 나누는 결정 경계를 찾는 것이
아니라, **두 클래스 사이의 마진(margin)을 가능한 한 크게 만드는 결정
경계**를 찾는 것이다.

``` text
클래스 A          결정 경계          클래스 B

● ● ●        |                ○ ○ ○
 ● ●         |                 ○ ○
             |
      ←------ margin ------→
```

결정 경계와 가장 가까운 훈련 샘플들이 모델의 결정에 중요한 역할을 하며,
이러한 샘플을 **서포트 벡터(Support Vector)**라고 한다.

------------------------------------------------------------------------

## 2. 선형 SVM 분류

### 2.1 라지 마진 분류

선형적으로 구분할 수 있는 데이터에서는 SVM이 두 클래스를 나누는 여러
직선 중 **가장 넓은 마진을 가지는 결정 경계**를 찾는다.

이를 **라지 마진 분류(Large Margin Classification)**라고 한다.

결정 경계에서 멀리 떨어진 샘플은 결정 경계에 거의 영향을 주지 않으며,
**마진의 경계에 위치한 서포트 벡터가 결정 경계를 결정**한다.

노트북에서는 Iris 데이터셋의 두 클래스를 사용해 선형 SVM의 결정 경계를
확인한다.

``` python
from sklearn.datasets import load_iris
from sklearn.svm import SVC

iris = load_iris(as_frame=True)

X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = iris.target

setosa_or_versicolor = (y == 0) | (y == 1)

X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]

svm_clf = SVC(kernel="linear", C=10**9)
svm_clf.fit(X, y)
```

`kernel="linear"`은 선형 결정 경계를 사용하는 SVM을 의미한다.

------------------------------------------------------------------------

## 2.2 특성 스케일의 중요성

### 이론

SVM은 **특성의 스케일에 매우 민감**하다.

한 특성의 값 범위가 다른 특성보다 훨씬 크면, SVM이 계산하는 거리에도 큰
영향을 주어 결정 경계가 좋지 않게 형성될 수 있다.

예를 들어 수직축의 값 범위가 수평축보다 훨씬 크다면 SVM이 한쪽 특성에
지나치게 영향을 받을 수 있다.

따라서 SVM을 사용할 때는 일반적으로 **특성 스케일링(feature scaling)**을
수행하는 것이 중요하다.

``` text
스케일링 전
특성 1 : 1 ~ 10
특성 2 : 1 ~ 1000

        ↓ StandardScaler

스케일링 후
두 특성이 비슷한 범위를 가짐
```

노트북에서는 `StandardScaler`를 이용해 특성을 표준화한다.

``` python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

스케일을 맞추면 SVM이 특성 간 거리를 더 적절하게 비교할 수 있어 **결정
경계가 훨씬 좋아질 수 있다.**

------------------------------------------------------------------------

## 2.3 소프트 마진 분류

### 이론

모든 훈련 샘플이 마진 밖에 있고 완벽하게 분류되도록 만드는 것을 **하드
마진 분류(Hard Margin Classification)**라고 한다.

하지만 하드 마진 분류에는 문제가 있다.

-   데이터가 선형적으로 완벽하게 구분되어야 한다.
-   이상치(outlier)에 매우 민감하다.

따라서 실제 문제에서는 일부 샘플이 마진 안에 들어오거나 잘못 분류되는
것을 허용할 필요가 있다.

이를 **소프트 마진 분류(Soft Margin Classification)**라고 한다.

소프트 마진 분류의 목표는

> **마진을 가능한 한 넓게 유지하는 것과 마진 오류를 줄이는 것 사이에서
> 적절한 균형을 찾는 것**

이다.

### C 하이퍼파라미터

SVM에서는 규제 하이퍼파라미터 **`C`**를 이용해 이 균형을 조절한다.

``` text
C가 작음
→ 규제가 강함
→ 더 넓은 마진
→ 더 많은 마진 오류 허용

C가 큼
→ 규제가 약함
→ 더 좁은 마진
→ 마진 오류를 더 강하게 제한
```

노트북에서는 `LinearSVC`와 스케일링을 파이프라인으로 묶어 사용한다.

``` python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

svm_clf = make_pipeline(
    StandardScaler(),
    LinearSVC(C=1, random_state=42)
)

svm_clf.fit(X, y)
```

확률 대신 결정 함수의 점수를 확인할 수 있다.

``` python
svm_clf.decision_function(X_new)
```

`LinearSVC`는 기본적으로 클래스 확률을 제공하지 않기 때문에
`predict_proba()` 대신 `decision_function()`을 사용할 수 있다.

------------------------------------------------------------------------

## 3. 비선형 SVM 분류

### 이론

실제 데이터에는 하나의 직선이나 초평면으로 구분할 수 없는 경우가 많다.

이런 **비선형 데이터셋**을 처리하는 한 가지 방법은 기존 특성에서 새로운
특성을 만들어 데이터의 차원을 확장하는 것이다.

예를 들어 특성 `x`만으로 데이터를 구분할 수 없다면 다음처럼 새로운
특성을 만들 수 있다.

``` text
기존 특성
x

        ↓

새로운 특성 추가
x, x²
```

특성을 확장한 공간에서는 원래 비선형적으로 보이던 데이터가 **선형적으로
구분 가능해질 수 있다.**

------------------------------------------------------------------------

## 3.1 다항 특성 추가

노트북에서는 `make_moons()`를 이용해 비선형 데이터셋을 생성한다.

``` python
from sklearn.datasets import make_moons

X, y = make_moons(
    n_samples=100,
    noise=0.15,
    random_state=42
)
```

이 데이터는 초승달 모양으로 분포되어 있어 하나의 직선으로 분류하기
어렵다.

`PolynomialFeatures`를 사용해 다항 특성을 추가할 수 있다.

``` python
from sklearn.preprocessing import PolynomialFeatures

polynomial_svm_clf = make_pipeline(
    PolynomialFeatures(degree=3),
    StandardScaler(),
    LinearSVC(C=10, max_iter=10_000, random_state=42)
)

polynomial_svm_clf.fit(X, y)
```

과정은 다음과 같다.

``` text
원본 비선형 데이터
        ↓
PolynomialFeatures
        ↓
고차원 특성 공간으로 변환
        ↓
StandardScaler
        ↓
LinearSVC
        ↓
비선형 결정 경계 표현
```

------------------------------------------------------------------------

## 3.2 다항식 커널

다항 특성을 직접 추가하면 낮은 차수에서는 잘 작동하지만, 차수가 커질수록
특성의 개수가 매우 많이 증가할 수 있다.

SVM에서는 **커널 트릭(kernel trick)**을 이용해 실제로 많은 특성을 직접
생성하지 않고도 고차원 특성을 추가한 것과 비슷한 효과를 얻을 수 있다.

노트북에서는 `SVC`의 다항식 커널을 사용한다.

``` python
from sklearn.svm import SVC

poly_kernel_svm_clf = make_pipeline(
    StandardScaler(),
    SVC(kernel="poly", degree=3, coef0=1, C=5)
)

poly_kernel_svm_clf.fit(X, y)
```

주요 하이퍼파라미터는 다음과 같다.

  하이퍼파라미터    의미
  ----------------- --------------------------------------
  `kernel="poly"`   다항식 커널 사용
  `degree`          다항식의 차수
  `coef0`           높은 차수와 낮은 차수 항의 영향 조절
  `C`               규제 강도 조절

모델이 과대적합된다면 다항식의 차수를 낮추는 방법을 고려할 수 있고,
과소적합이라면 차수를 높일 수 있다.

------------------------------------------------------------------------

## 3.3 유사도 특성과 RBF 커널

비선형 문제를 해결하는 또 다른 방법은 각 샘플이 특정
**랜드마크(landmark)**와 얼마나 비슷한지를 측정해 새로운 특성을 만드는
것이다.

노트북에서는 **가우스 RBF(Gaussian Radial Basis Function)** 개념을
다룬다.

RBF를 이용하면 랜드마크와 가까운 샘플은 높은 유사도를, 멀리 있는 샘플은
낮은 유사도를 갖게 할 수 있다.

실제로 모든 랜드마크에 대한 특성을 직접 만들면 계산량이 커질 수 있으므로
SVM에서는 **RBF 커널**을 사용할 수 있다.

``` python
rbf_kernel_svm_clf = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", gamma=5, C=0.001)
)

rbf_kernel_svm_clf.fit(X, y)
```

### gamma

RBF 커널에서 중요한 하이퍼파라미터가 **`gamma`**이다.

``` text
gamma 증가
→ 각 샘플의 영향 범위가 좁아짐
→ 결정 경계가 복잡해질 수 있음

gamma 감소
→ 각 샘플의 영향 범위가 넓어짐
→ 결정 경계가 부드러워짐
```

따라서 모델이 과대적합되면 `gamma`나 `C`를 낮추고, 과소적합되면 높이는
방향을 고려할 수 있다.

### 어떤 커널을 사용할까?

일반적으로 먼저 **선형 커널**을 시도할 수 있다.

훈련 세트가 너무 크지 않다면 **RBF 커널**도 많이 사용되며 다양한
데이터셋에서 잘 작동한다.

------------------------------------------------------------------------

## 4. SVM 회귀

### 이론

SVM은 분류뿐만 아니라 **회귀(Regression)**에도 사용할 수 있다.

분류에서는 목표가

> 두 클래스 사이에 가능한 한 **넓은 도로(마진)**를 만드는 것

이었다.

회귀에서는 목표를 반대로 생각한다.

> 제한된 마진 오류 안에서 가능한 한 **많은 샘플이 도로 안에 들어오도록**
> 학습한다.

``` text
SVM 분류

클래스 A   |      넓은 마진      |   클래스 B
● ● ●      |                     |    ○ ○ ○


SVM 회귀

       ┌───────────────────────┐
       │ ●  ●   ● ●   ●       │
-------│------ 예측 함수 -------│-------
       │   ●   ●    ●          │
       └───────────────────────┘
             ε 마진
```

------------------------------------------------------------------------

## 4.1 선형 SVM 회귀

사이킷런에서는 선형 SVM 회귀를 위해 `LinearSVR`을 사용할 수 있다.

``` python
from sklearn.svm import LinearSVR

svm_reg = make_pipeline(
    StandardScaler(),
    LinearSVR(
        epsilon=0.5,
        random_state=42
    )
)

svm_reg.fit(X, y)
```

### epsilon

SVM 회귀에서는 **`epsilon`**이 도로의 폭을 결정한다.

``` text
epsilon 증가
→ 도로가 넓어짐

epsilon 감소
→ 도로가 좁아짐
```

도로 안에 샘플이 추가되더라도 모델의 예측에 영향을 주지 않는 특성을
**ε-민감하지 않다(epsilon-insensitive)**고 한다.

------------------------------------------------------------------------

## 4.2 비선형 SVM 회귀

비선형 회귀 문제에서는 `SVR`에 커널을 적용할 수 있다.

노트북에서는 다항식 커널을 사용한다.

``` python
from sklearn.svm import SVR

svm_poly_reg = make_pipeline(
    StandardScaler(),
    SVR(
        kernel="poly",
        degree=2,
        C=0.01,
        epsilon=0.1
    )
)

svm_poly_reg.fit(X, y)
```

`SVR`의 주요 하이퍼파라미터는 다음과 같다.

  하이퍼파라미터   역할
  ---------------- --------------------
  `kernel`         사용할 커널 결정
  `degree`         다항식 커널의 차수
  `C`              규제 강도 조절
  `epsilon`        마진의 폭 결정

`C`가 작을수록 규제가 강해지고, 더 많은 샘플이 마진 밖에 위치하는 것을
허용할 수 있다.

------------------------------------------------------------------------

## 5. SVM의 핵심 원리

SVM 분류기는 결정 함수를 이용해 새로운 샘플의 클래스를 예측한다.

선형 SVM에서는 입력 특성과 학습된 가중치를 이용해 결정 함수 값을
계산하고, 그 값의 부호에 따라 클래스를 결정한다.

``` text
결정 함수 값 < 0 → 한 클래스
결정 함수 값 > 0 → 다른 클래스
```

결정 경계는 결정 함수 값이 `0`이 되는 지점이다.

### 마진

결정 경계와 가장 가까운 샘플들이 **서포트 벡터**이며, 이 샘플들이 마진의
위치를 결정한다.

따라서 SVM은 모든 훈련 샘플이 결정 경계에 동일하게 영향을 주는 것이
아니라 **경계 근처의 샘플이 특히 중요한 모델**이다.

### 커널 트릭

비선형 SVM의 핵심은 **커널 트릭**이다.

``` text
원래 공간에서는
선형 분리 불가능

        ↓ 커널

더 높은 차원의 특성 공간에서는
선형 분리 가능한 것처럼 계산

        ↓

원래 공간에서는
비선형 결정 경계 형성
```

이를 통해 명시적으로 매우 많은 고차원 특성을 생성하지 않고도 복잡한
비선형 결정 경계를 학습할 수 있다.

------------------------------------------------------------------------
#질문 
