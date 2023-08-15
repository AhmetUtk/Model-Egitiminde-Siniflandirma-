Merhaba , Bu projede model eğitiminde hedef kolondaki değerlerin değişkenliği fazlalaştığında neler yapılabilir onunla uğraştım. Kod kısmından önce size biraz veri setiyle ilgili bilgi vereyim. Veri seti şarapların sitrik asit oranı , pH değeri , alkol oranı , kalite vs. gibi özelliklerini tutmaktadır. Bizim bu projede hedeflediğimiz şarapların özelliklerinden kalite tahmini yapabilecek bir yapay sinir ağı oluşturmaktı. Şimdi de kodlarken öğrendiğim ve tecrübe ettiğim aşamalardan bahsedeyim.

**********************************************

Öncelikle main.py dosyasında binary yani ikili sınıflandırma uygulamaya çalıştım ve quality kolonundaki değerlerden 5 ten fazla olanları 1 az olanları 0 olarak atamaya çalıştım ve modeli eğittim daha sonra sınıflandırma parametresi olarak 5 yerine 6 yı veya 7 yi seçip modeli eğitince modelin daha fazla doğruluk oranı verdiğini gözlemledim. Bunun sebebinin 5 ten büyük değerlerin sayısının 6 dan veya 7 den büyük değerlerin sayısından daha çok olduğu için 6 veya 7 yapınca modelin öğrenmesinin daha kolaylaştığını farkettim ve yorumlayabildim.

**********************************************

Daha sonra main1.py dosyasında multi yani çoklu sınıflandırma yapmaya çalıştım bunun için hedef kolondaki yani quality kolonundaki değerleri 3 ile 5 arasında olanları 0, 5 ile 8 arasında olanları 1, 8 ve 8 den büyük olanları 2 olarak sınıflandıracak bir siniflandir adlı bir metot yazdım ve veri setine uyguladım çoklu sınıflandırma ile düzenlediğim veri seti üzeirnde eğitim yaptığımda ikili sınıflandırmaya göre daha doğru çalıştığını gözlemledim.

***********************************************

En son aşamada ise main2.py dosyasında araştırmalarım sonucunda benim yapmak istediğim sınıflandırmayı yapacak bir fonksiyon olduğunu öğrendim bu foksiyon keras.utils kütüphanesinin içinde bulunan to_categorical fonksiyonudur. Bu fonksiyon sayısal değerleri kategorikleştirerek modelin sınıfları daha iyi şekilde anlamasına yardımcı oluyor.

***********************************************

NOT : Proje içinde aynı veri setini 3 defa koydum çünkü her dosyada veri seti üzerinde değişim yapılıyor. Kodu çalıtırıp veri seti üzerinde değişim yaptıktan sonra sınıflandırıcı kısımlarını silebilir veya aynı veri setini kopyalayıp tekrar kullanabilirsiniz.

