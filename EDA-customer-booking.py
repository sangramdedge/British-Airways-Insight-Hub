#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis on Customer Bookings data for British Airways

# In[1]:


#imports

import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


import pandas as pd

# Use the correct file path directly
file_path = "C:/Users/sangr/Downloads/customer_booking.csv"

# Read the CSV file
df = pd.read_csv(file_path, encoding="ISO-8859-1")


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.describe()


# In[7]:


df.info()


# # Sales Channel

# In[8]:


per_internet = df.sales_channel.value_counts().values[0]  / df.sales_channel.count() *100
per_mobile = df.sales_channel.value_counts().values[1]  / df.sales_channel.count() *100


# Number of Bookings 

# In[9]:


print(f"Number of bookings done through internet: {per_internet} %")
print(f"Number of bookings done through phone call: {per_mobile} %")


# Trip Type

# In[10]:


per_round = df.trip_type.value_counts().values[0]/ df.trip_type.count() *100
per_oneway = df.trip_type.value_counts().values[1]/ df.trip_type.count() *100
per_circle = df.trip_type.value_counts().values[2]/ df.trip_type.count() *100


# In[11]:


print(f"Percentage of round trips: {per_round} %")
print(f"Percentage of One way trips: {per_oneway} %")
print(f"Percentage of circle trips: {per_circle} %")


# Purchase Lead

# In[12]:


plt.figure(figsize=(15,5))
sns.histplot(data=df, x="purchase_lead", binwidth=20,kde=True)


# There are few bookings that were done more than 2 years before the travel date and it seems very unlikely that book that in advance. However, it might also be because of the cancellation and rebooking in a period of 6 months for twice. Generally airline keep the tickets for rebooking within a year. But at this point we will consider them as outliers which will effect the results of predictive model in a huge way.

# In[13]:


(df.purchase_lead >600).value_counts()


# In[14]:


df[df.purchase_lead > 600]


# In[15]:


#filtering the data to have only purchase lead days less than 600 days
df = df[df.purchase_lead <600 ]


# length of stay

# In[16]:


plt.figure(figsize=(15,5))
sns.histplot(data=df, x="length_of_stay", binwidth=15,kde=True)


# In[17]:


#Let's see how many entries do we have that exceeds length of stay more than 100 days.

(df.length_of_stay> 200).value_counts()


# In[18]:


df[df.length_of_stay> 500].booking_complete.value_counts()


# In[19]:


#filtering the data to have only length of stay days less than 500 days
df = df[df.purchase_lead <500 ]


# Flight Day

# In[20]:


mapping = {
    "Mon" : 1,
    "Tue" : 2,
    "Wed" : 3,
    "Thu" : 4,
    "Fri" : 5,
    "Sat" : 6,
    "Sun" : 7
}

df.flight_day = df.flight_day.map(mapping)


# In[21]:


df.flight_day.value_counts()


# In[22]:


#Booking Origin


# In[23]:


plt.figure(figsize=(15,5))
ax = df.booking_origin.value_counts()[:20].plot(kind="bar")
ax.set_xlabel("Countries")
ax.set_ylabel("Number of bookings")


# In[24]:


#Above chart shows travellers from which country had maximum booking applications.


# In[25]:


plt.figure(figsize=(15,5))
ax = df[df.booking_complete ==1].booking_origin.value_counts()[:20].plot(kind="bar")
ax.set_xlabel("Countries")
ax.set_ylabel("Number of complete bookings")


# In[27]:


#Above chart shows travellers from which country had their booking complete.


# In[28]:


##Booking complete


# In[30]:


successful_booking_per = df.booking_complete.value_counts().values[0] / len(df) * 100


# In[31]:


unsuccessful_booking_per = 100-successful_booking_per


# In[32]:


print(f"Out of 50000 booking entries only {round(unsuccessful_booking_per,2)} % bookings were successfull or complete.")


# In[ ]:





# In[34]:


# Export Data


# In[33]:


df.to_csv(cwd + "/filtered_customer_booking.csv")


# In[ ]:




