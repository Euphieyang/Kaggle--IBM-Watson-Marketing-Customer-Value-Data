#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


data=pd.read_csv('final_data.csv')


# ## 1. Describe your dataset

# In[3]:


data
#汽車保險客戶的檔案
#我們要處理的y: customer lifetime value(用戶在其生命週期內所能帶來的商業價值)
#這仍是行銷data
#具有許多缺失值，可把這些row刪掉


# In[4]:


data.head()
#具有許多缺失值，可把這些row刪掉


# In[5]:


data.shape
#共有9134位customer的資料，共有24欄


# In[6]:


data.info()
#customer: id
#state: 所在州
#clv: 
#response: 回報事故
#coverage: 哪種保險類型
#Education
#Effective To Date: 到期時效
#EmploymentStatus: 工作狀態
#Gender: 性別(可去深入研究)
#Income: 收入
#Location Code: 市區/郊區
#Marital Status: 婚姻狀態
#Monthly Premium Auto: 最貴保費自動轉帳(可深入研究)
#Months Since Last Claim: 多久之前申請保險
#Months Since Policy Inception: 上次保險的生效日
#Number of Open Complaints:客戶抱怨的次數
#Number of Policies:屬於哪種保險
#Policy Type:保險種類(公司/個人)
#Policy:保險內容
#Renew Offer Type:更新內容(使用代碼去代表)
#Sales Channel:怎麼買保險(打電話/跟業務員買)
#Total Claim Amount:總共賠償多少錢
#Vehicle Class :汽車種類
#Vehicle Size :汽車大小


# # 2.Data-preprocessing

# #### (1)處理缺失值

# In[7]:


data.Customer.nunique()
#有8099筆Customer沒有重複、不是缺失值


# In[8]:


#了解缺失值概況
#發現從8099之後的顧客紀錄皆為NaN
data[data.Customer.isnull()]


# In[9]:


#刪除缺失值
data.dropna(axis=0,inplace=True)


# In[10]:


#再次確認是否有缺失值
data.isnull().sum()


# In[11]:


data
#customer: id
#state: 所在州
#clv: 
#response: 回報事故
#coverage: 哪種保險類型
#Education
#Effective To Date: 到期時效
#EmploymentStatus: 工作狀態
#Gender: 性別(可去深入研究)
#Income: 收入
#Location Code: 市區/郊區
#Marital Status: 婚姻狀態
#Monthly Premium Auto: 最貴保費自動轉帳(可深入研究)
#Months Since Last Claim: 多久之前申請保險
#Months Since Policy Inception: 上次保險的生效日
#Number of Open Complaints:客戶抱怨的次數
#Number of Policies:屬於哪種保險
#Policy Type:保險種類(公司/個人)
#Policy:保險內容
#Renew Offer Type:更新內容(使用代碼去代表)
#Sales Channel:怎麼買保險(打電話/跟業務員買)
#Total Claim Amount:總共賠償多少錢
#Vehicle Class :汽車種類
#Vehicle Size :汽車大小


# #### (2)重複值檢查

# In[12]:


x=data.duplicated()
x.value_counts()
#沒有重複


# #### (3)資料類型轉換

# In[13]:


data.head()


# In[14]:


#確認目前Columns的型態
#把該換的column換成數字、時間
#需轉換為數字: nope
#需轉換為時間: Effective To Date 
#需轉換為虛擬變數:state, response, coverage, education, 
#EmploymentStatus, gender, policy type, policy, 
#Renew Offer Type, sales channel, Vehicle Class, Vehicle Size   

data.dtypes


# In[15]:


#先將有空白的名字作處理
data.rename(columns={'Months Since Policy Inception':'months_since_policy_inception','EmploymentStatus':'Employment_Status','Customer Lifetime Value':'customer_lifetime_value', 'Effective To Date':'effective_to_date','Location Code':'location_code','Marital Status':'marital_status','Monthly Premium Auto':'monthly_premium_auto','Months Since Last Claim':'months_since_last_claim','Number of Open Complaints':'number_of_open_complaints','Number of Policies':'number_of_policies','Policy Type':'policy_type','Renew Offer Type':'renew_offer_type','Sales Channel':'sales_channel','Total Claim Amount':'total_claim_amount','Vehicle Class':'vehicle_class','Vehicle Size':'vehicle_size'},inplace = True)


# In[16]:


data.columns


# In[17]:


data['validated_date']=pd.to_datetime(data.effective_to_date)


# In[18]:


data.info()


# In[19]:


#虛擬變數
data2=pd.get_dummies(data, columns=['marital_status','location_code','State', 'Response', 'Coverage', 'Education','Employment_Status', 'Gender', 'policy_type', 'Policy','renew_offer_type', 'sales_channel', 'vehicle_class','vehicle_size'], drop_first=True)


# In[20]:


#確認有沒有設立成功
data2.info()


# In[165]:


#不知道為什麼要先跑一張圖autoviz才會出現
channel=data['sales_channel'].value_counts()
s2=pd.Series(channel)

ax2=s2.plot.pie(autopct='%.1f')

ax2.set_title('sales_channel')
ax2.set_ylabel('')


fig1=ax2.figure
fig1.set_size_inches(8,3)
fig1.tight_layout(pad=1)
plt.show()


# # 3.Visualization

# In[22]:


get_ipython().system('pip install autoviz')


# In[23]:


from autoviz.AutoViz_Class import AutoViz_Class
Vis=AutoViz_Class()


# In[24]:


data2.to_csv('data2.csv')


# In[181]:


Vis_1=Vis.AutoViz('data2.csv')


# # 4. Find the important relationship between variables

# In[26]:


data2.corr()


# In[27]:


#CLV和其他變數相關性
data2.corr()['customer_lifetime_value']


# In[28]:


#賠償金額與其他變數的相關性
data2.corr()['total_claim_amount']


# In[29]:


data2.corr()['vehicle_class_Luxury Car']


# In[30]:


data2.corr()['vehicle_class_Luxury SUV']


# # 5. Find three business insights

# In[31]:


data.describe()


# ### 1. 銷售管道

# In[32]:


#銷售管道&保險類型
channel_coverage=data.groupby(['sales_channel']).Coverage.value_counts()


# In[33]:


channel_coverage=channel_coverage.unstack()


# In[34]:


channel_coverage_plot=channel_coverage.plot(kind='bar',figsize=(10,7))
channel_coverage_plot.set_ylabel('the number of the coverage')

plt.show()


# ### 2. 調整保險內容

# In[35]:


#保險內容更新
renew=data.groupby('Policy').renew_offer_type.value_counts()
renew


# In[36]:


renew=renew.unstack()


# In[180]:


renew_plot=renew.plot(kind='bar',figsize=(10,7))
renew_plot.set_ylabel('the number of the renew')

plt.show()


# ### 3. 保險公司主要顧客群

# In[162]:


#依據州劃分
state_location=data.groupby('State').location_code.value_counts()


# In[163]:


state_location=state_location.unstack()
state_location.plot(kind='bar',figsize=(10,7),ylabel='the number of people')


# In[170]:


#依據居住類型劃分，了解保險類型及汽車種類對平均賠償金額
suburban=data[data['location_code']=='Suburban']
suburban


# In[173]:


suburban.groupby(['Coverage','vehicle_class']).customer_lifetime_value.mean()


# In[177]:


employment=suburban.groupby(['Employment_Status']).total_claim_amount.mean()


# In[179]:


employment.plot(kind='bar',figsize=(10,7),ylabel='average claim')


# ### 4.主要客戶: 前25% 顧客終身價值的被保人

# In[38]:


#了解前25%被保人概況
potential_customer=data[data['customer_lifetime_value']>8963.294993]
potential_customer


# In[39]:


#前25%被保人申報次數
potential_response=potential_customer['Response'].value_counts()
r=pd.Series(potential_response)

response=r.plot.pie(autopct='%.1f')

response.set_title('Response')
response.set_xlabel('')
response.set_ylabel('')

fig=response.figure
fig.set_size_inches(8,3)
fig.tight_layout(pad=1)
plt.show()


# In[40]:


#婚姻狀態
potential_marital=potential_customer['marital_status'].value_counts()
m=pd.Series(potential_marital)

marital=m.plot.pie(autopct='%.1f')

marital.set_title('marital_status')
marital.set_xlabel('')
marital.set_ylabel('')

fig=marital.figure
fig.set_size_inches(8,3)
fig.tight_layout(pad=1)
plt.show()


# In[41]:


#前25%被保人的保險類型
potential_coverage=potential_customer['Coverage'].value_counts()
c=pd.Series(potential_coverage)

coverage=c.plot.pie(autopct='%.1f')

coverage.set_title('the amount of coverage')
coverage.set_xlabel('')
coverage.set_ylabel('')

fig=coverage.figure
fig.set_size_inches(8,3)
fig.tight_layout(pad=1)
plt.show()


# In[42]:


#前25%被保人的婚姻狀態及教育程度
potential_customer_marital=potential_customer.groupby(['marital_status','Education']).customer_lifetime_value.mean()


# In[43]:


potential_customer_marital=potential_customer_marital.unstack()
potential_customer_marital.plot(kind='bar',figsize=(10,7), ylabel='the average of value')
plt.show()


# In[44]:


#前25%被保人保險類型及汽車種類
potential_customer_coverage_car=potential_customer.groupby(['Coverage','vehicle_class']).customer_lifetime_value.mean()


# In[45]:


potential_customer_coverage_car=potential_customer_coverage_car.unstack()


# In[46]:


potential_customer_coverage_car.plot(kind='bar',figsize=(10,7), ylabel='the average of value')
plt.show()


# In[161]:


potential_customer.corr()['monthly_premium_auto']


# In[ ]:





# In[ ]:





# In[ ]:




