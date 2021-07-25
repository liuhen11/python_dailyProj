# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2020/5/24 20:54
# @Author : Mat
# @Email : mat_wu@163.com
# @File : spider2.py
# @Software: PyCharm

import requests
from lxml import etree
import time
import win32com.client
speak = win32com.client.Dispatch('SAPI.SPVOICE')

prox = {
    'http':'http://182.106.136.69:63482'

}
def parse():
    for i in range(1,150):
        url = 'https://lawnew.cnki.net/kns/brief/brief.aspx?curpage=1&RecordsPerPage=50&QueryID=62&ID=&turnpage=1&tpagemode=L&dbPrefix=CLKLK&Fields=&DisplayMode=listmode#J_ORDER'.format(i)
        headers = {
            'Sec-Fetch-Dest': 'iframe',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://kns.cnki.net/kns/brief/brief.aspx?pagename=ASP.brief_result_aspx&isinEn=0&dbPrefix=CLKD&dbCatalog=%e4%b8%ad%e5%9b%bd%e6%b3%95%e5%be%8b%e7%9f%a5%e8%af%86%e8%b5%84%e6%ba%90%e6%80%bb%e5%ba%93&ConfigFile=CLKD.xml&research=off&t=1621767906804&keyValue=&S=1&sorttype=',
            'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
            'Cookie':'Ecp_ClientId=3210425151503214069; cnkiUserKey=74fc14cf-beb0-17df-a48d-6fd281a06d83; Ecp_ClientIp=117.136.64.76; Ecp_session=1; SID=012055; ASP.NET_SessionId=2o3wxx3ikkkllbltyib1kk5c; CurTop10KeyWord=null%2C%u4EA7%u4E1A; RsPerPage=50; Ecp_loginuserbk=xn0185; _pk_ref=%5B%22%22%2C%22%22%2C1622169834%2C%22https%3A%2F%2Fwww.baidu.com%2Flink%3Furl%3DgkgoQE3euj2t6amkA6luVtqDGN18YbRt-6_FqIuwT8G%26wd%3D%26eqid%3D9ecd11dc0001131f0000000660afa71b%22%5D; _pk_id=655845d3-013e-4f08-9e20-7b4483b301f3.1619334956.11.1622169834.1622169834.; Ecp_LoginStuts={"IsAutoLogin":false,"UserName":"xn0185","ShowName":"%e5%9b%9b%e5%b7%9d%e5%a4%a7%e5%ad%a6%e9%94%a6%e5%9f%8e%e5%ad%a6%e9%99%a2%e5%9b%be%e4%b9%a6%e9%a6%86","UserType":"bk","BUserName":"","BShowName":"","BUserType":"","r":"mDxfFk"}; LID=WEEvREcwSlJHSldSdmVqeVpQU2FpY3FZSnpUUUx1UlpVS253a0dYWWlpbz0=$9A4hF_YAuvQ5obgVAqNKPCYcEjKensW4IQMovwHtwkF4VYPoHbKxJw!!'
        }

        res = requests.get(url,headers=headers,proxies = prox).text
        print(res)
        e = etree.HTML(res)
        if '验证码：' in res:
            print(i)
            #speak.Speak('你该休息了！')
            break
        else:
            table = e.xpath('//table[@class="GridTableContent"]/tr')
            for g in table[1:]:
                item = {}
                cc = g.xpath('./td[2]/a//text()')[0].replace('document.write(ReplaceChar1(ReplaceChar(ReplaceJiankuohao(',                                             '')
                print(cc)
                cc1 = cc.replace('<font class=Mark>', '')
                cc2 = cc1.replace('</font>', '')
                cc3 = cc2.replace('))));', '')
                item['title'] = ''.join(cc3)
                item['from'] = ''.join(g.xpath('./td[3]/text()')).strip()
                item['date'] = ''.join(g.xpath('./td[4]/text()')).strip()
                item['db'] = ''.join(g.xpath('./td[5]/text()')).strip()
                item['level'] = ''.join('地方政府规章及文件').strip()
                with open('zhiwang6.json','a+',encoding='utf-8') as fp:
                    fp.write(str(item) + '\n')
                time.sleep(1)
        time.sleep(5)


if __name__ == '__main__':
    parse()