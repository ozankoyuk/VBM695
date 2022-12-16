## Hacettepe Üniversitesi Veri ve Bilgi Mühendisliği Yüksek Lisans Programı <br /> [VBM 695 Bitirme Projesi]

### Projenin Konusu
  Türkiye'nin 2018 yılından günümüze kadarki günlük ve saatlik elektrik tüketim verilerini, derin öğrenme algoritmaları ile birlikte eğiterek/kullanarak gelecek gün ve saatler için tüketim tahmini üretilmesi ve gerçekleşen değerler ile karşılaştırılması ve elde edilmesi beklenen kar ve zarar miktarlarının gösterimi.
  
  Hedefim, eğitim verisi olarak bir fabrikanın veya her gün EPİAŞ üzerinden [Gün Öncesi Piyasası (**GÖP**)](https://seffaflik.epias.com.tr/transparency/piyasalar/gop/ptf.xhtml) aracılığı ile bir sonraki gün için elektrik kullanımı satın alanların geçmiş tüketim verilerini kullanarak, bir sonraki gün için en doğru miktarda elektrik tüketim miktarlarını ve elde edilmesi beklenen kar-zarar durumunu göstermektir.
 
 --- 
### Kullanılan Teknolojiler ve Algoritmalar
* [Python 3.10.6](https://www.python.org/doc/)
* [Long-Short Term Memory (LSTM)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
* [Autoregressive Integrated Moving Average (ARIMA)](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average),
* [Prophet](https://facebook.github.io/prophet/)

---
### Yol Haritası
* Projemin yol haritasına [ilgili bağlantıdan](https://github.com/ozankoyuk/VBM695/blob/main/Ozan%20K%C3%B6y%C3%BCk%20Yol%20Haritas%C4%B1.pdf) erişebilirsiniz.

---
### Kurulum
* Projenin tamamı Python 3.10.6 sürümü geliştirilmiştir. Bu sebepten dolayı tavsiye edilen sürüm **3.6.10** sürümüdür.
* Eğer daha önceden yüklenmediyse, aşağıdaki komut ile Python 3.10.6 için venv modülü yüklenir
```
  sudo apt install python3.10-venv
```
* Daha sonra `python3 -m venv venv` komutu ile bir sanal ortam yaratılır ve `. venv/bin/activate` komutu ile aktif hale getirilir.
* Sanal ortamda çalıştırılacak `pip -V` komutu ile sürüm kontrolü yapılır, eğer aşağıdaki sürüm **22.0.2** sürümünden eski ise `python3 -m pip install --upgrade pip` komutu ile son sürüme güncellenebilir.
```
pip 22.0.2 from [PATH]/venv/lib/python3.10/site-packages/pip (python 3.10)
```
* Tüm sürümlerin güncel ve sistemin hazır olduğunu teyit ettikten sonra `pip install -r requirements.txt` komutu ile projenin çalışması için gerekli tüm kütüphaneler, sabitlenen sürümleri ile indirilir.
* İndirme işlerinin tamamlanmasından sonra `python main.py` komutunun çalıştırılması tüm algoritmaların çalışıp sonuçların çıkmasını sağlayacaktır. 

### Ek bilgiler
* [crawler.py](https://github.com/ozankoyuk/VBM695/blob/main/crawler.py) dosyası, içinde belirtilen tarih aralığınıa ait verilerin Şeffaflık platformundan çekilmesini sağlamaktadır. Tüm dosyalar JSON formatında [yıl_ay.json] olarak kaydedilmektedir.
* [converter.py](https://github.com/ozankoyuk/VBM695/blob/main/converter.py) dosyası, bu klasörler altındaki dosyaları sıra ile açıp bir dataframe içerine aktarır. Tüketim ve tahmin verileri ayrı ayrı uçlardan indirildiği için, bu dosya ile tüm bu veriler tek bir dataframe haline getirilir ve CSV formatında kaydedilir.
* Tüm bu adımlar halihazırda tamamlanmıştır. Tek yapılması gereken [main.py](https://github.com/ozankoyuk/VBM695/blob/main/main.py) dosyasının çalıştırılmasıdır.
* Farklı tarih aralığında veriler kullanılmak istenirse, indirilen tüm veriler silinmelidir: [tahmin klasörü](https://github.com/ozankoyuk/VBM695/tree/main/next_day_pred) ve [gerçekleşen tüketim klasörü](https://github.com/ozankoyuk/VBM695/tree/main/real_consumption)
