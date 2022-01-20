# Kmeans-Clustering
TEORI
CLUSTERING DATA
Clustering adalah salah satu teknik dari algoritma machine learning yaitu unsupervised learning. Algoritma clustering membagi populasi atau data point dengan sifat yang sama  ke beberapa kelompok kecil untuk dikelompokkan. K-means clustering merupakan salah satu metode cluster analysis non hirarki yang berusaha untuk mempartisi objek yang ada kedalam satu atau lebih cluster atau kelompok objek berdasarkan karakteristiknya, sehingga objek yang mempunyai karakteristik yang sama dikelompokan dalam satu cluster yang sama dan objek yang mempunyai karakteristik yang berbeda dikelompokan kedalam cluster yang lain. Metode K-Means Clustering berusaha mengelompokkan data yang ada ke dalam beberapa kelompok, dimana data dalam satu kelompok mempunyai karakteristik yang sama satu sama lainnya dan mempunyai karakteristik yang berbeda dengan data yang ada di dalam kelompok yang lain.
Contoh penggunaan clustering :
 










DATA
Clustering data Pelanggaran prokes dan Angka Kematian Covid19
Pada Bulan Oktober 2021 di Jawa Tengah
1.	Tujuan
Clustering data 
2.	Input
Alasan memilih input adalah peneliti ingin mengetahui hasil pengelompokan pada 2 data yang salig berdampak, yaitu pelanggaran prokes dan kematian karena covid19, dipilih variabel input :
•	Angka pelanggaran prokes
•	Angka kematian karena covid19
3.	Definisi Input
•	Angka pelanggaran prokes
Angka pelanggaran prokes didapat dari sheet 5 tugas dengan judul “pelanggaran protkes” 
•	Angka kematian karena covid19
Angka kematian tiap kota/kabupaten disesuaikan dengan data pelanggaran prokes, data didapatkan dari sheet 2 tugas dengan juduk “Covid19”, data setiap kota  diolah dari menjumlahkan setiap kabupaten. 
Misal : untuk mendapatkan data kematian kota/kabupaten Wonosobo maka diperlukan data kematian pada kabupaten seperti berikut :
Tabel 1 Data kematian per kota/kabupaten
 
Untuk kota/kabupaten yang lain mempunyai perhitungan yang sama.
Dari perhitungan diatas maka diperoleh data yang akan diolah menggunakan google colab yaitu sebagai berikut :
Tabel 2 Data Pelanggaran prokes dan kematian akibat COVID19
 


PERHITUNGAN DENGAN GOOGLE COLAB
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
df=pd.read_csv("/content/Book1.csv")
df.head()
plt.scatter(df.Pelanggaran,df['Kematian'])
plt.xlabel('Pelanggaran')
plt.ylabel('Kematian')
km=KMeans(n_clusters=3)
y_predicted=km.fit_predict(df[['Pelanggaran','Kematian']])
y_predicted
df['cluster']=y_predicted
df.head()
df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]
plt.scatter(df1.Pelanggaran,df1['Kematian'],color='green')
plt.scatter(df2.Pelanggaran,df2['Kematian'],color='red')
plt.scatter(df3.Pelanggaran,df3['Kematian'],color='blue')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='black',marker='*',label='centroid')
plt.xlabel('Pelanggaran')
plt.ylabel('Kematian')
plt.legend()
sse=[]
k_rng=range(1,10)
for k in k_rng:
  km=KMeans(n_clusters=k)
  km.fit(df[['Pelanggaran','Kematian']])
  sse.append(km.inertia_)
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)

Dari program didapatkan hasil clustering yaitu :
 
Gambar 1 Hasil Clustering
 
Gambar 2 Cluster optimal



ANALISIS HASIL
1.	Menurut Gambar 2 maka diperoleh bahwa pembagian cluster yang paling optimal untuk data adalah 3, hal itu ditunjukkan dari grafik elbow (siku yang menunjuk tepat di angka 3), sehingga pembagian kelompok data yang digunakan adalah dibagi menjadi 3 kelompok cluster.
2.	Menurut Gambar 1 maka diperoleh :
Cluster 1: Pada cluster 1 (Green/hijau) diperoleh bahwa 2 data pada bulan Oktober tahun 2021 merupakan kelompok angka kematian tinggi dan angka pelanggaran rendah
Cluster 2: Pada cluster 2 (Red/merah) diperoleh bahwa terdapat 29 data pada bulan Oktober tahun 2021 merupakan kelompok angka kematian rendah dan angka pelanggaran rendah
Cluster 3: Pada cluster 3 (Blue/biru) diperoleh bahwa 4 data pada bulan Oktober tahun 2021 merupakan kelompok angka kematian rendah dan angka pelanggaran tinggi
3.	Pada Gambar 1 pusat cluster adalah tanda bintang, perhitungan menggunakan centroid cluster, sehingga data yang tersebar disekitar pusat cluster merupakan data yang memiliki karakteristik yang sama dan dapat dikatan menjadi 1 kelompok
Pada Gambar 1 diketahui bahwa cluster 2 (Red/merah) diperoleh terdapat 29 data (mayoritas data disbanding cluster lainnya) pada bulan Oktober tahun 2021 merupakan kelompok angka kematian rendah dan angka pelanggaran rendah, hal tersebut mengindikasikan bahwa terdapat hubungan positif antara angka kematian dan angka pelanggaran prokes, dimana jika angka pelanggaran prokes rendah maka angka kematian cenderung rendah.
