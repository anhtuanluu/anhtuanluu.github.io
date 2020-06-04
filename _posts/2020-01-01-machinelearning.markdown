---
layout: post
title:  "Top các thuật toán machine Learning "
date:   2020-01-01 21:03:36 +0530
categories: AI, python
---
Bài viết này sẽ như là 1 tour đơn giản giới thiệu 1 vòng về cuộc hành trình trong khoa học về dữ liệu và machine learning. Xuyên qua bài viêt này, bạn sẽ có thể làm việc với các vấn đề trong các thuật toán của machine learning với code Python.  
Bài viết có lược bỏ các phần có liên quan đến toán thống kê cho dễ hiểu 😄 .  

![](https://user-images.githubusercontent.com/66369791/83756512-58e79300-a699-11ea-83a8-9ca7d385a372.jpg)  

Đầu tiên, bạn cần nắm rõ.  

## Có 3 loại thuật toán trong Machine Learning  

Dưới đây mình sẽ nói về tư tưởng của từng thuật toán.  
  
### 1. Supervised Learning  
  
Cách thức vận hành : Đây là thuật toát bao gồm một biến mục tiêu / kết quả (hoặc biến phụ thuộc). biến này được dự đoán từ 1 tập hợp các nhân tố ( biến độc lập) . Sử dụng những tập hợp của các biến, chúng ta tạo được một hàm có thể cho phép các kết quả output đúng với dự đoán tương ứng với kết quả input. Quá trình "training" này sẽ tiếp tục cho đến các chúng ta đạt được một độ chính xác mong muốn liên quan đến việc "traing dữ liệu".  

Ví dụ của các thuật toán Supervised Learning : Regression, Decision Tree, Random Forest, KNN, Logistic Regression v.v.v…  

### 2. Unsupervised Learning  

Cách thức vận hành : Trong thuật toán này, chúng ta không có bất kì biến mục tiêu / kết quả để dự đoán / đánh giá . Nó được sử dụng để phân nhóm các tập dữ liệu vào các nhóm khác nhau. Các nhóm này được sử dụng cho việc phân khúc các "khách hàng' vào các nhóm khác nhau nữa cho mục tiêu cụ thể.  

Ví dụ của Unsupervised Learning : thuật toán Apriori, K-means.  

### 3. Reinforcement Learning  

Cách thức vận hành : Sử dụng thuật toán này, máy sẽ được luyện tập để ra các quyết định. Máy sẽ được tiếp xúc với một môi trường nơi mà nó được tự luyện tập liên tục sử dụng với phương thức "thử và sửa sai" cho đến khi "sai" đạt đến ngưỡng chấp nhận được hoặc là sẽ bằng 0 trong điều kiện tuyệt đối. Máy sẽ học từ các "kinh nghiệm trong quá khứ" và "học các kiến thức" tốt nhất có thể được để tạo nên các quyết định chính xác về mặt logic business.  

Ví dụ của Reinforcement Learning: Markov Decision Process  

Dưới đây là danh sách các thuật toán Machine Learning phổ biến. Những thuật toán này có thể được áp dung cho bất kì vấn đề nào về dữ liệu :  
  
   Linear Regression  
   Logistic Regression  
   Decision Tree  
   SVM  
   Naive Bayes  
   KNN  
   K-Means  
   Random Forest  
   Dimensionality Reduction Algorithms  
   Gradient Boost & Adaboost  

## 1. Linear Regression  

Line Regression được sử dụng để ước tính giá trị thực (giá nhà, số lượng cuộc gọi, doanh thu bán hàng v.v.v.. dựa trên các biến thay đổi liên tục). Ở đây, chúng ta thiết lập mối quan hệ giữa các biến độc lập và biến phục thuộc bằng cách lắp 1 line tốt nhất, line này được biết đến như một dòng đệ qui và được biểu diễn bằng công thức Y = a*X + b  

Cách tốt nhất để hiểu linear regression là nhớ laị kinh nghiệm tuổi thơ. Hãy xem, bạn yêu cầu một đứa bé lớp 5 sắp mọi người của nó trong lớp bằng thứ tự tăng dần cân năng mà không hỏi cả lớp về cân năng gì cả. Đứa bé sẽ làm gì ? Khả năng cao nó sẽ nhìn vào chiều cao và kết cấu cơ thể để sắp xếp cả lớp sử dụng sự kết hợp các thông số mà nó có thể nhìn thấy. Đây chính là Linear Regression trong đời sống thực. Đứa bé thực tế đã quan sát được chiều và kết cấu cơ thể sẽ có sự tương quan với trọng lượng, đây chính là điều giống như công thức ở trên.  

Trong công thức trên :  

Y - Biến phục thuộc  

a - Slope  

X - Biến độc lập  

b - Intercept  

Các hệ số a và b đều bắt nguồn từ việc giảm thiếu các khoảng chênh lệch bình phương giữa khoảng cách của các điểm dữ liệu và regression line.  

Tham khảo ví dụ dưới đây. Ở đây chúng ta đã xác định rõ được line tốt nhất có công thức là y=0.2811x+13.9. Bây giờ, sử dụng công thức này, chúng ta có thể tìm được cân năng, nếu chúng ta biết được chiều cao của một người nào đó.  

![](https://user-images.githubusercontent.com/66369791/83757282-72d5a580-a69a-11ea-8bf8-066e09c43984.png)  

Linear Regression được phân làm 2 loại chính : Simple Linear Regression và Multiple Linear Regression. Simple Linear Regression được đặc trưng bởi một biến đọc lập. Multiple Linear Regression được đặc trung bởi nhiều hơn 1 biến độc lập. Trong khi tìm kiếm line tốt nhất, bạn có thể sẽ tìm được một hôi qui đa thức hoặc đường cong.  

Code ví dụ:  

```python
#Import Library
#Import other necessary libraries like pandas, numpy...
from sklearn import linear_model

#Load Train and Test datasets
#Identify feature and response variable(s) and values must be numeric and numpy arrays
x_train=input_variables_values_training_datasets
y_train=target_variables_values_training_datasets
x_test=input_variables_values_test_datasets

# Create linear regression object
linear = linear_model.LinearRegression()

# Train the model using the training sets and check score
linear.fit(x_train, y_train)
linear.score(x_train, y_train)

#Equation coefficient and Intercept
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
#Predict Output
predicted= linear.predict(x_test)
```  

## 2. Logistic Regression  

Đừng nhầm lẫn bởi tên của nó ! Đây là một sự phân loại, khoogn phải là một thuật toán hồi qui. Nó được sử dụng để ddanhs giá ước tính các giá trị ời rạc (các giá trị nhị phân) dựa trên tập hợp các biến độc lập. Một cách đơn giản, nó dự đoán xác suất xảy ra của một sử kiện bằng việc khớp dữ liệu với một hàm logit. Do đó, nó còn được gọi là hồi qui logit. Và cũng vì lì do vậy, nó dự đoán xác suất, output của nó luôn nằm giữa 0 và 1.  

Một lần nữa, chúng ta hãy cố gắng hiểu thuật toán này thông qua một ví dụ đơn giản .  

Hãy giả sử bạn của bạn cho bạn 1 bài toàn để giải. Chỉ có 2 kịch bản kết quả - bản giải được or ko giải được. Bây giờ thử tưởng tượng rằng ban đang được đưa 1 giải các câu đố/toán/quizz để có thể hiểu được bản thân là bạn phù hợp với types câu đố nào. Kết quả của việc nghiên cứu bản thân này sẽ trông như thế này - nếu bạn được đưa một vấn đề kiểu giải đố toán lớp 10, bạn có 70% xác suất giải được nó. Nếu nó là câu hỏi lịch sử của lớp 5, xác suất chỉ còn 30%. Đây chính là Logistic Regression.  

Quay trở về với toán, các tỷ lệ log kết quả được mô hình hóa bằng sự kết hợp tuyến tính của các biến dự đoán.  


