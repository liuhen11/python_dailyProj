# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2020/5/24 21:09
# @Author : Mat
# @Email : mat_wu@163.com
# @File : spider3.py
# @Software: PyCharm

import requests
from lxml import etree

url_dict = {}
url_dict['地方司法文件'] = 'https://lawnew.cnki.net/kns/brief/brief.aspx?dbprefix=CLKLK&queryID=56&clusField=%E6%95%88%E5%8A%9B%E7%BA%A7%E5%88%AB%E4%BB%A3%E7%A0%81&clusCode=15&Param=&t=1621862580802'
headers1 = {
            'Sec-Fetch-Dest': 'iframe',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://lawnew.cnki.net/kns/brief/result.aspx?dbPrefix=CLKLK',
            'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36 Edg/90.0.818.66',
            'Cookie':'Ecp_ClientId=3210425151503214069; cnkiUserKey=74fc14cf-beb0-17df-a48d-6fd281a06d83; Ecp_ClientIp=117.136.64.76; Ecp_session=1; Ecp_loginuserbk=KT1005; SID=012055; ASP.NET_SessionId=2o3wxx3ikkkllbltyib1kk5c; CurTop10KeyWord=null%2C%u4EA7%u4E1A; RsPerPage=50; _pk_ref=%5B%22%22%2C%22%22%2C1621856075%2C%22https%3A%2F%2Fwww.baidu.com%2Flink%3Furl%3DfW0Fg6nBiC5dcdGuQ_f5_HfJLjOoMR91s-7T7bvKLvy%26wd%3D%26eqid%3Db01b90c300013c0b0000000660ab81f3%22%5D; _pk_id=655845d3-013e-4f08-9e20-7b4483b301f3.1619334956.4.1621856168.1621856075.; Ecp_LoginStuts={"IsAutoLogin":false,"UserName":"KT1005","ShowName":"%e8%a5%bf%e5%8d%97%e4%ba%a4%e9%80%9a%e5%a4%a7%e5%ad%a6","UserType":"bk","BUserName":"","BShowName":"","BUserType":"","r":"Q3SXe0"}; LID=WEEvREcwSlJHSldSdmVqeVpQU2FpY3FVMDBmL3k4bW9KZmRNeXRNZmRPZz0=$9A4hF_YAuvQ5obgVAqNKPCYcEjKensW4IQMovwHtwkF4VYPoHbKxJw!!'
        }

res = requests.get(url_dict['地方司法文件'],headers=headers1).text
print(res)
e = etree.HTML(res)
table = e.xpath('//table[@class="GridTableContent"]/tr')
for i in table[1:]:
    item = {}
    cc = i.xpath('./td[2]/a//text()')[0].replace('document.write(ReplaceChar1(ReplaceChar(ReplaceJiankuohao(', '')
    print(cc)
    cc1 = cc.replace('<font class=Mark>','')
    cc2 = cc1.replace('</font>','')
    cc3 = cc2.replace('))));','')
    item['title'] = ''.join(cc3)
    item['from'] = ''.join(i.xpath('./td[3]/text()')).strip()
    item['date'] = ''.join(i.xpath('./td[4]/text()')).strip()
    item['db'] = ''.join(i.xpath('./td[5]/text()')).strip()
    item['level'] = ''.join('地方司法文件').strip()
    print(item)
    with open('zhiwang2.json','a+',encoding='utf-8') as fp:
        fp.write(str(item) + '\n')