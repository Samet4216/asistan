---
baslik: Semantik Arama (Semantic Search)
tarih: 2026-09-01
durum: aktif
etiketler: [yapay-zeka, arama, vektör, teknofest]
---
# Semantik Arama (Semantic Search)

Semantik arama (anlamsal arama), kelimelerin birebir eşleşmesine bakmak yerine, aranan cümlenin veya kelimenin "anlamını" kavrayarak sonuç getiren modern arama teknolojisidir.

## Nasıl Çalışır?
Geleneksel aramalar (Keyword Search) metin içinde kelimesi kelimesine eşleşme arar. Örneğin "araba" aratırsan, içinde sadece "araba" geçen yerleri bulur.

Semantik arama ise kelimeleri karmaşık sayılara (**vektörlere / embeddings**) çevirir. "Araba" kelimesi ile "Otomobil" veya "Araç" kelimelerinin matematiksel uzaydaki konumları birbirine çok yakındır. Bu sayede, sen "araba" diye arattığında sistem, metinde sadece "otomobil" geçse bile sana o sonucu getirebilir, çünkü kelimelerin **anlamsal yakınlığını** bilir.

## RAG ile İlişkisi
[[rag-sistemi]] mimarisinin tam kalbinde Semantik Arama yatar. 
RAG sisteminde kullanıcı yapay zekaya bir soru sorduğunda (Retrieval aşamasında), sistem önce bu sorunun anlamını vektörlere çevirir ve veritabanında "anlamsal olarak en yakın" dokümanları bulur. 

## Teknofest Projen İçin Neden Önemli?
Eğer projende kullanıcının sorduğu bir soruya dokümanlardan cevap bulman gerekiyorsa, klasik arama (Elasticsearch vb.) yerine vektör veritabanları (ChromaDB, Pinecone, Milvus vb.) kullanarak semantik arama yapmalısın. Bu, projenin çok daha "akıllı" ve insan gibi anlamasını sağlar.
