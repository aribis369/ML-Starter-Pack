from sklearn.svm import SVR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


db=pd.read_csv("/home/arindam/Desktop/googl.csv")
nb=db.values
print(nb[0])

def con_date(a):
    mon={"Jan":"1","Feb":"2","Mar":"3","Apr":"4","May":"5","Jun":"6","Jul":"7","Aug":"8","Sep":"9","Oct":"10","Nov":"11","Dec":"12"}

    for d in a:   
        date=d[0].split("-")
        if int(date[0])>=10 and int(mon[date[1]])>=10:
            new_date=str(date[2]+mon[date[1]]+date[0])
        elif int(mon[date[1]])>=10 and int(date[0])<10:
            new_date=str(date[2]+mon[date[1]]+"0"+date[0])
        elif int(mon[date[1]])<10 and int(date[0])>=10:
            new_date=str(date[2]+"0"+mon[date[1]]+date[0])
        else:
            new_date=str(date[2]+"0"+mon[date[1]]+"0"+date[0])            
        n_date=int(new_date)
        d[0]=n_date
  

#con_date(nb)
dates=[]
open_st=[]
clo_st=[]
k=0
for d in nb:
    dates.append(k)
    open_st.append(float(d[1]))
    clo_st.append(float(d[4]))
    k+=1

dates=np.reshape(dates,(len(dates),1))
open_st=np.reshape(open_st,(len(open_st),1))
clo_st=np.reshape(clo_st,(len(clo_st),1))

print(dates)
print(open_st)

rbf_open=SVR(kernel="rbf", C=1e3, gamma=0.1)
rbf_clo=SVR(kernel="rbf", C=1e3, gamma=0.1)

rbf_open.fit(dates,open_st)
rbf_clo.fit(dates,clo_st)

#plt.scatter(dates,open_st,color="black",label="data")
plt.plot(dates,rbf_open.predict(dates), color="red", label="Reg")
plt.plot(dates,rbf_clo.predict(dates), color="green", label="Reg")
plt.xlabel("Date")
plt.ylabel("Price")

plt.legend()
plt.show()

#https://github.com/search?utf8=%E2%9C%93&q=kossiitkgp&type=

#*/10 * * * * export DISPLAY=:0; /usr/bin/python3 /home/arindam/Desktop/movies.py > /home/arindam/Desktop/listener.log 2>&1



