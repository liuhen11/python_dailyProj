# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2020/5/24 20:54
# @Author : Mat
# @Email : mat_wu@163.com
# @File : spider2.py
# @Software: PyCharm

import requests
from lxml import etree


def parse():
    for i in range(1,3):
        url = 'https://lawnew.cnki.net/kns/brief/brief.aspx?curpage={}&RecordsPerPage=50&QueryID=24&ID=&turnpage=1&tpagemode=L&dbPrefix=CLKLK&Fields=&DisplayMode=listmode#J_ORDER'.format(i)
        headers = {
            'Sec-Fetch-Dest': 'iframe',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://kns.cnki.net/kns/brief/brief.aspx?pagename=ASP.brief_result_aspx&isinEn=0&dbPrefix=CLKD&dbCatalog=%e4%b8%ad%e5%9b%bd%e6%b3%95%e5%be%8b%e7%9f%a5%e8%af%86%e8%b5%84%e6%ba%90%e6%80%bb%e5%ba%93&ConfigFile=CLKD.xml&research=off&t=1621767906804&keyValue=&S=1&sorttype=',
            'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
            'Cookie':'Ecp_ClientId=3210425151503214069; cnkiUserKey=74fc14cf-beb0-17df-a48d-6fd281a06d83; Ecp_ClientIp=117.136.64.76; Ecp_session=1; Ecp_loginuserbk=KT1005; SID=012055; ASP.NET_SessionId=2o3wxx3ikkkllbltyib1kk5c; CurTop10KeyWord=null%2C%u4EA7%u4E1A; RsPerPage=50; _pk_ref=%5B%22%22%2C%22%22%2C1621856075%2C%22https%3A%2F%2Fwww.baidu.com%2Flink%3Furl%3DfW0Fg6nBiC5dcdGuQ_f5_HfJLjOoMR91s-7T7bvKLvy%26wd%3D%26eqid%3Db01b90c300013c0b0000000660ab81f3%22%5D; _pk_id=655845d3-013e-4f08-9e20-7b4483b301f3.1619334956.4.1621856168.1621856075.'
        }

        res = requests.get(url,headers=headers).text
        print(res)
        e = etree.HTML(res)
        table = e.xpath('//table[@class="GridTableContent"]/tr')
        for i in table[1:]:
            item = {}
            cc = i.xpath('./td[2]/a//text()')[0].replace('document.write(ReplaceChar1(ReplaceChar(ReplaceJiankuohao(',                                             '')
            print(cc)
            cc1 = cc.replace('<font class=Mark>', '')
            cc2 = cc1.replace('</font>', '')
            cc3 = cc2.replace('))));', '')
            item['title'] = ''.join(cc3)
            item['from'] = ''.join(i.xpath('./td[3]/text()')).strip()
            item['date'] = ''.join(i.xpath('./td[4]/text()')).strip()
            item['db'] = ''.join(i.xpath('./td[5]/text()')).strip()
            item['level'] = ''.join('地方性法规及文件').strip()
            with open('zhiwang2.json','a+',encoding='utf-8') as fp:
                fp.write(str(item) + '\n')


if __name__ == '__main__':
    parse()