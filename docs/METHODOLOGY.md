# Yöntem ve değerlendirme

Tarihsel akış: yaklaşık 3.000 manuel tweet → lojistik regresyonla diğer tweetleri etiketleme → karma etiketli CSV üzerinde DistilBERTurk fine-tuning → Streamlit. Model etiketleri insan doğrulaması olarak kabul edilmez.

Mevcut yerel eğitim notebook'u DistilBERTurk, 128 token, 3 epoch ve rastgele 80/10/10 bölümleme kullanıyor. Notebook'ta pozitif sınıf için yaklaşık 0,979 F1 görülüyor. Karma etiketli testteki bu değer gerçek çağrılar üzerinde bağımsız doğruluk kanıtı değildir; öğretmen modelin etiketleriyle uyumu da içerebilir. CV'de bu metrik kullanılmamıştır. Notebook çıktıları kamuya aktarılmamıştır.

Yeni baseline TF-IDF (kelime 1–2, karakter 3–5) ve scikit-learn LogisticRegression kullanır. Tekil normalize metinler rastgele 80/20, stratified ve seed=42 ile ayrılır. TF-IDF yalnızca eğitim bölümünde öğrenilir. Test yalnızca manuel etiketlerden oluşur; sentetik demo açıkça ayrı işaretlenir. Çelişen kopyalar eğitimden önce reddedilir. Sınıf 1 için karar eşiği 0,5'tir; başka eşik yalnızca ayrıca ayrılmış doğrulama kümesinde seçilmelidir.

Rastgele metin bölümlemesi yakın kopyaları, olay veya yazar ilişkisini bütünüyle çözmez. Gelecekte olay/yazar grupları ve zamana göre test; bağımsız manuel kontrol; precision/recall, PR-AUC ve hata analizi gerekir. Başarı veya kurtarma etkisi garantisi verilmez.

Bu çalışma araştırma portföyüdür; otomatik acil durum önceliklendirmesi, gerçeklik doğrulaması veya spam tespitinin tam çözümü değildir.
