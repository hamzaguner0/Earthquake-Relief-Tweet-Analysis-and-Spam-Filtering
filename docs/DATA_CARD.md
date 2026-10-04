# Veri kartı ve paylaşım kararı

## Kamuya açık içerik

Bu depoda kod, yöntem açıklaması ve özgün sentetik örnekler paylaşılır. Gerçek tweet CSV'leri, yanlış tahmin listeleri, kullanıcı bilgileri, model ağırlıkları ve notebook çıktıları yayımlanmaz. Kaggle'a gerçek veri yükleme yapılmamıştır.

## Tarihsel yerel veri

Kullanıcı beyanı: yaklaşık 3.000 manuel etiket, ardından lojistik regresyonla model etiketleme. Özgün internet kaynağının URL'si, lisansı ve satır düzeyinde etiket kökeni mevcut dosyalardan doğrulanamadı. Birleştirilmiş CSV bağımsız ground truth değildir. Yanlış pozitif/negatif dosyaları da özgün metin içerdikleri için özel tutulur.

Etiketler: 1=yardım çağrısı adayı, 0=diğer. Bir çağrının gerçekliğini veya insanın güvenliğini doğrulayan etiketler değildir.

## Kişisel veri

Telefon, kullanıcı adı, bağlantı, ad/soyad ve ayrıntılı konum kişileri tanımlayabilir. Sağlık bilgileri özel nitelikli kişisel veri kapsamına girebilir. Çocuklarla ilgili içerik ve acil yardım koşulları ek hassasiyet taşır. Bir kişinin açık paylaşım yapmış olması, başka amaçlarla sınırsız işleme/yayma izni değildir. [KVKK alenileştirme açıklaması](https://www.kvkk.gov.tr/Icerik/6843/-ALENILESTIRME-HAKKINDA-KAMUOYU-DUYURUSU), [özel nitelikli veriler](https://www.kvkk.gov.tr/Icerik/2051/Ozel-Nitelikli-Kisisel-Veriler).

Telefonları veya kullanıcı adlarını maskelemek tek başına anonimlik sağlamaz; adres, bağlam ve diğer kaynaklar kişiyi yeniden belirleyebilir. Anonimlik başka verilerle eşleştirildiğinde de kişiye bağlanamama gerektirir. [KVKK anonimleştirme](https://www.kvkk.gov.tr/Icerik/2038/kisisel-verilerin-silinmesi-yok-edilmesi-veya-anonim-hale-getirilmesi).

## Kaynak koşulları

4 Ekim 2026 tarihinde kontrol edilen [X geliştirici yönergeleri](https://docs.x.com/developer-guidelines), içerik toplama, yeniden dağıtım ve AI/ML eğitimi için kısıtlar içerir; güncel metin Grok dışındaki model eğitimini yasaklayan bir hüküm içeriyor. Tarihsel toplama tarihindeki şartlar, verinin elde edildiği kaynak ve lisansı ayrıca değerlendirilmelidir. Mevcut kod yayınlamak bu hakları sağlamaz; anonimlik de lisans yerine geçmez. Tweet ID paylaşımı veya Kaggle yüklemesi kendiliğinden izinli sayılmaz.

## Güvenli devam

Özgün elle etiketlenen alt küme ve toplama kaydı geri bulunur; izinler doğrulanır; kopyalar ve çelişkiler elle incelenir; kaynak kökeni korunur. İzin doğrulanamıyorsa özgün sentetik örnekler veya kullanımına açıkça izin veren başka veri tercih edilir. Gerçek veri ve ondan öğrenilmiş ağırlıklar paylaşım değerlendirmesi bitene kadar yerelde kalır.

Denetim düzenli ifade ipuçlarından toplu sayılar üretir. Özel nitelikli verilerin tam tespiti, lisans incelemesi veya anonimlik garantisi değildir.
