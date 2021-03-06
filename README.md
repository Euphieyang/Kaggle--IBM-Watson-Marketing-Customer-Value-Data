# Kaggle: IBM Watson Marketing Customer Value Data

## 1. Describe the dataset
這是一份保險客戶的檔案，我們負責的是行銷部分，著重在Customer lifetime value(CLV)最大化<br>
將資料載入後，發現共有9134筆顧客保險資料，並有24欄變數項目，其後半部分的資料，竟然呈現缺失狀態，且同一欄之中，具有多種類別，再去細看此檔案的資訊，發現其type部分為object，部分為float

## 2.	Data-preprocessing
* Unique value<br>
先確認這份檔案之中，是否有重複的顧客資料<br>
發現顧客資料並無重複，因此並不用執行資料刪除的動作<br>

* 	Missing value<br>
由於第一步驟--觀察資料時，發現後半部分呈現NaN<br>
因此先了解檔案之中的缺失值數，以及相應位置，接著執行刪除缺失值，最後確認檔案<br>

* Type <br>
  1.	時間<br>
  細看變數，觀察到有一欄位是屬於時間，然而其type呈現object<br>
  因此先透過類型轉換，將這變數轉換成datetime64，最後再確認是否有轉換成功<br>

  2.	虛擬變數<br>
  另一方面，這份資料中具有多項變數為object類型，然而同一欄位之中存在不同的類別<br>
  我們可以知道，這是個名目測量尺度(nominal scale)<br>因此採用虛擬變數，以利我們分析<br>
  設立完虛擬變數之後，最後確認我們是否有成功設立<br>
  
  
## 3.	Data Visualization
透過autoviz的幫助，將資料一次性全部轉換為圖表，可得36張圖<br>
在這之中，我們看出一些端倪，例如: <br>

1. 顧客居住在不同地區，平均賠償金額明顯呈現不同<br>
2. 不同款車型所創造的CLV、自動Premium方案轉帳金額明顯不同<br>
3. 不同婚姻狀態下，單身者平均賠償金額明顯高於離婚及已婚者<br>

## 4. Relationships between Variables
從圖表中，能看到不同種類的保險、汽車種類、婚姻狀態對於CLV呈現出明顯不同，因此推測變數具有相關性，因此分別一一判斷變數項目是否存在重要關聯<br>

1.	CLV<br>
從相關係數表得知，採用Premium方案自動轉帳、總賠償金額、保險專案為Premium，及車子為Luxury系列為正相關，而以自動轉帳金額的相關性最高。<br>

2.	賠償金額<br>
從相關係數表中觀看，得知自動轉帳付款金額越高的駕駛，其賠償金額較高<br>
推測可能是因為其保險方案是較高等級的關係，因此保險公司所賠償的金額才會如此高<br>
另一方面居住在郊區的居民，賠償金額相對於居住在城市或鄉下的人較高昂，相關性高達0.6。<br>

3.	車款<br>
車子種類與自動轉帳金額、收入、總賠償金額呈現正相關性<br>
以luxury系列車款與賠償金額較相關性較高，推估可能是因為其本身車款維修費用較高，抑或是其車禍嚴重程度高的關係<br>

## 5. Find three business insights
1.	強化網站銷售管道<br>
   我們可以從圖知道，不管是哪種方案，大眾仍較多是透過Agent申請保險，其次為Branch，且以Basic最多，Premium方案為最少人申請，而且以網站申請Premium方案，次數都是最少的。<br>

    可推估，當大眾簽訂保單時，大眾會傾向找尋Agent來為自己服務，以尋得信任及安心感，因此公司須加強Agent的人員培訓。<br>
    除此之外，若公司希望能透過Web管道，增加公司行政效率，減少Call Center人事上成本的話，或許應考慮多向民眾宣傳，使用Web方式申請，並搭配獎勵活動，提高申辦意願，例如: 贈送虛擬點數，兌換連鎖餐廳小餐點、邀請好友申請，可獲得100元好友獎勵。<br>
![Image](https://github.com/Euphieyang/Kaggle--IBM-Watson-Marketing-Customer-Value-Data/blob/main/5-1.jpg)
   
2. 調整既有方案內容
   從統計資料中，可得知不同方案類型的保戶，他們最多人選擇更新的保險內容為offer1，推估此內容吸引保戶，保險公司可考慮將offer1內容加入至既有保險方案之中，以吸引目前非保戶族群購買保險<br>
   ![Image](https://github.com/Euphieyang/Kaggle--IBM-Watson-Marketing-Customer-Value-Data/blob/main/5-2.png)

3. 保險公司主要顧客群
   從圖表可知，保險公司的顧客明顯分布於郊區，且根據autoviz所呈現各居住類型中，郊區的平均賠償金額較高<br>
   推測都市居民因為較長搭乘公共交通設施，因此較少擁有汽車，而不保車險，對於鄉村地區，則是因為道路較寬廣，發生車禍事故較少的緣故，賠償金額並不高，然而住在郊區的顧客可能對於用車需求大，所以車險的需求較高，因此才會呈現資料中佔大多數來自郊區<br>
   
   原先推估是因為居住在郊區的居民，上班地點在都市的關係，需時常往返兩地，而提高其發生嚴重車禍，因此賠償金額較高的緣故<br>
   然而卻發現賠償金額高的保戶，其職業狀態是失業，保險公司可能需要去深入了解大部分失業保戶的車禍原因主要是為何。<br>
 ![Image](https://github.com/Euphieyang/Kaggle--IBM-Watson-Marketing-Customer-Value-Data/blob/main/5-3.jpg)
   
4. 保險公司前端20%顧客
   依據 80/20法則，公司大部分的利潤來自於前端部分的顧客，因此先透過前25%顧客終身價值的客戶，了解這間保險公司主要保戶樣貌，我們可知前25%具潛力性的被保人，共有2025位，其中58.8%為已婚狀態，其次為25.6%單身者，最後是15.7%的離婚者<br>
   在已婚被保人之中，以高中程度(含)以下的人是最多，其次為Bachelor，Doctor是最少人<br>

   從資料集中仔細觀察，高達84.7%駕駛並未申報保險賠償，推測他們皆無發生車禍<br>
   除此之外，能為保險公司創造平均淨收益的顧客佔大多數為Extended 保險類型，駕駛Luxury系列的車主
   而較特別的是，保premium方案的Sports car顧客，所創造的淨收益大，因此保險公司可考慮透過搭售的方式，與Sports car、Luxury車款廠商合作，向顧客推銷Extend及Premium方案<br>

   另一點，值得提的是，前端保戶中，其premium方案自動轉帳的金額與總賠償金額，呈現將近0.7的正相關性
   ![Image](https://github.com/Euphieyang/Kaggle--IBM-Watson-Marketing-Customer-Value-Data/blob/main/5-4.jpg)
