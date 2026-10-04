# Deprem Yardım Çağrılarının Sınıflandırılması

Türkçe deprem paylaşımlarındaki yardım çağrısı adaylarını diğer içerikten ayırmaya yönelik kişisel NLP araştırması. **Operasyonel acil yardım sistemi değildir.** Metinden bir çağrının gerçekliğini doğrulamak veya otomatik kurtarma önceliği belirlemek mümkün değildir.

## Projenin geçmişi ve katkım

İnternette bulunan paylaşımlardan yaklaşık **3.000 tweeti elle sınıflandırdım**. Bu etiketler üzerinden lojistik regresyonla diğer metinlere model etiketi ürettim. Daha sonra DistilBERTurk fine-tuning ve Streamlit arayüzü üzerinde çalıştım.

Yerel tarihsel CSV, elle ve modelle üretilen etiketleri satır düzeyinde ayırt eden güvenilir bir kaynak sütunu içermiyor. Bu nedenle geçmiş notebook sonuçları bağımsız, elle etiketlenmiş bir test kümesi başarısı olarak sunulmuyor. Özgün lojistik regresyon kodu mevcut dosyalarda bulunmadığı için bu depodaki TF-IDF + LogisticRegression akışı **yeniden kurulmuş bir baseline**dır. Algoritmayı sıfırdan yazdığıma ilişkin bir iddia içermez; scikit-learn kullanır.

## Bu depoda ne var?

- Eğitim verisi içinde `manual`, `pseudo`, `synthetic` köken ayrımı; kaynağı belirsiz etiketlerle eğitim reddedilir.
- Aynı normalize metnin eğitim ve testte bulunmasını engelleyen kontrol; çelişen etiketler elle inceleme gerektirir.
- Eğitim bölümünde öğrenilen TF-IDF ve lojistik regresyon; yalnızca ayrılmış etiketli test üzerinde değerlendirme.
- Yerel, toplu veri denetimi; rapora tweet metni, telefon veya kullanıcı adı yazılmaz.
- Yerel modelle çalışan Streamlit demosu; tarihsel DistilBERTurk için isteğe bağlı yerel yükleme.
- Tamamı bu depo için yazılmış sentetik kurulum örnekleri ve kontroller.

Gerçek tweetler, hata örnekleri, eğitilmiş gerçek model ağırlıkları ve özel dosyalar bu depoda yayımlanmaz. Tarihsel iki notebook yalnızca geçmiş denemeleri göstermek için tutulur; üretim akışı değildir. Veri toplama notebook'u devreden çıkarılmıştır.

## Kurulum ve sentetik demo

Python 3.10+ (yerel doğrulama: Python 3.12).

```sh
git clone https://github.com/hamzaguner0/Earthquake-Relief-Tweet-Analysis-and-Spam-Filtering.git
cd Earthquake-Relief-Tweet-Analysis-and-Spam-Filtering
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[app]"
python -m unittest discover -s tests -v
deprem-nlp train examples/synthetic_demo.csv --demo
deprem-nlp predict "Sentetik örnek: enkaz altında mahsur kaldık yardım gerekiyor"
streamlit run app.py
```

Sentetik metrikler kurulumun çalıştığını gösterir; gerçek deprem verisi başarısını ölçmez. Oluşturulan `artifacts/` klasörü Git dışında tutulur. `joblib` dosyalarını yalnızca kendi oluşturduğunuz güvenilir kaynaktan yükleyin.

## İzinli gerçek veride çalışma

Önce kaynak, toplama tarihi, lisans/platform şartları ve kişisel veri işleme koşulları doğrulanmalıdır. `--authorized-data` bu doğrulama yapıldıktan sonra kullanılan bir beyan seçeneğidir; izin kazandırmaz.

CSV: `text,label,label_source` (0=diğer, 1=yardım çağrısı; baseline için tamamı `manual`). Kaynağı belirsiz tarihsel satırlara topluca `manual` yazmayın. Özgün 3.000 satırın kimlikleri/etiket kayıtları geri bulunmalı; bulunamayan kayıtlar yeniden etiketlenmelidir.

```sh
deprem-nlp audit private/manual.csv
deprem-nlp train private/manual.csv --authorized-data
deprem-nlp pseudo-label private/unlabeled.csv --authorized-data
```

Yeni model etiketleri `pseudo` olarak kaydedilir. Eğitim ve test metinleri yeni etiketleme çıktısından çıkarılır. Bu baseline pseudo etiketleri eğitim/teste katmaz. Yakın kopyalar, aynı olay ve aynı yazardan gelen kayıtlar için ayrıca gruplu veya zamana göre bağımsız bir test gerekir.

Yerel DistilBERTurk için `pip install -e ".[app,transformer]"`; `fine_tuned_distilberturk/` klasörü veya `DEPREM_MODEL_DIR` ortam değişkeni kullanılır. Tarihsel model en fazla 128 token ile çalışır; sınıf 1 eşlemesi yerel eğitimdeki yardım çağrısı etiketine dayanır. Bu model yeniden eğitilmedi ve güncel bağımsız başarısı doğrulanmadı.

## Veri ve paylaşım

[Veri kartı ve paylaşım kararı](docs/DATA_CARD.md) · [Yöntem ve değerlendirme sınırları](docs/METHODOLOGY.md)

Kod paylaşımı, üçüncü taraf tweetleri yeniden dağıtma hakkı vermez. Sentetik örnekler gerçek tweetlerden türetilmemiştir. Gerçek veri için genel bir açık veri lisansı tanımlanmamıştır.
