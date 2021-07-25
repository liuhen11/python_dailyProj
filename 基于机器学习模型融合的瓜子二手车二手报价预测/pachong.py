
import requests
from lxml import etree
import time
import pandas as pd

#pinpai_url = 'https://www.guazi.com/cd/dazhong/#bread'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36 SLBrowser/7.0.0.4071 SLBChan/30',
           'Cookie': 'uuid=b8bc7fd0-5e7f-4516-9149-024e11120706; clueSourceCode=%2A%2300; user_city_id=45; Hm_lvt_bf3ee5b290ce731c7a4ce7a617256354=1619266325; ganji_uuid=8033529849733161868170; sessionid=31e3512c-7a58-4775-d30b-bda8558d0223; lg=1; close_finance_popup=2021-04-24; lng_lat=104.05356_30.70195; gps_type=1; _gl_tracker=%7B%22ca_source%22%3A%22-%22%2C%22ca_name%22%3A%22-%22%2C%22ca_kw%22%3A%22-%22%2C%22ca_id%22%3A%22-%22%2C%22ca_s%22%3A%22self%22%2C%22ca_n%22%3A%22-%22%2C%22ca_i%22%3A%22-%22%2C%22sid%22%3A72758527944%7D; track_id=201793344713297920; isTouFangGuaziIndex=1; guazitrackersessioncadata=%7B%22ca_kw%22%3A%22%25e7%2593%259c%25e5%25ad%2590%25e4%25ba%258c%25e6%2589%258b%25e8%25bd%25a6%22%7D; cainfo=%7B%22ca_a%22%3A%22-%22%2C%22ca_b%22%3A%22-%22%2C%22ca_s%22%3A%22seo_baidu%22%2C%22ca_n%22%3A%22default%22%2C%22ca_medium%22%3A%22-%22%2C%22ca_term%22%3A%22%25E7%2593%259C%25E5%25AD%2590%25E4%25BA%258C%25E6%2589%258B%25E8%25BD%25A6%25E6%2588%2590%25E4%25BA%25A4%25E8%25AE%25B0%25E5%25BD%2595%22%2C%22ca_content%22%3A%22%22%2C%22ca_campaign%22%3A%22%22%2C%22ca_kw%22%3A%22%25e7%2593%259c%25e5%25ad%2590%25e4%25ba%258c%25e6%2589%258b%25e8%25bd%25a6%22%2C%22ca_i%22%3A%22-%22%2C%22scode%22%3A%22-%22%2C%22keyword%22%3A%22-%22%2C%22ca_keywordid%22%3A%22184883397078%22%2C%22ca_transid%22%3A%22%22%2C%22platform%22%3A%221%22%2C%22version%22%3A1%2C%22track_id%22%3A%22201793344713297920%22%2C%22guid%22%3A%22b8bc7fd0-5e7f-4516-9149-024e11120706%22%2C%22ca_city%22%3A%22cd%22%2C%22display_finance_flag%22%3A%22-%22%2C%22client_ab%22%3A%22-%22%2C%22sessionid%22%3A%2231e3512c-7a58-4775-d30b-bda8558d0223%22%7D; cityDomain=cd; preTime=%7B%22last%22%3A1619266951%2C%22this%22%3A1619266323%2C%22pre%22%3A1619266323%7D; Hm_lpvt_bf3ee5b290ce731c7a4ce7a617256354=1619266952'}


def single_pinpai(pinpai_url,headers):
    resp = requests.get(pinpai_url,headers=headers)
    text = resp.content.decode('utf-8')
    html = etree.HTML(text)

    ul = html.xpath('//ul[@class="carlist clearfix js-top"]')[0]

    # detail_urls = []
    # name_Pinpais = []
    # price_Sellers = []
    # price_newGuides = []
    # #date_Password = []
    # year_Login_ins = []
    # car_Kilometers = []
    # car_Pailiangs = []
    # car_Gearboxs = []
    lis = ul.xpath('./li')
    for li in lis:
        try:
            name_Pinpai = li.xpath('./a/h2[@class="t"]')[0].text
            #name_Pinpais.append(name_Pinpai)
            year_Login_in = li.xpath('./a/div[@class="t-i"]')[0].text
            #year_Login_ins.append(year_Login_in)
            price_Seller = li.xpath('./a/div[@class="t-price"]/p/text()')[0]
            #price_Sellers.append(price_Seller)
            price_newGuide = li.xpath('./a/div[@class="t-price"]/em[@class="line-through"]')[0].text.split('万')[0]
            #price_newGuides.append(price_newGuide)

            detail_url = li.xpath('./a/@href')
            detail_url = 'https://www.guazi.com'+detail_url[0]

            resp_detail = requests.get(detail_url, headers=headers)
            text_detail = resp_detail.content.decode('utf-8')
            html_detail = etree.HTML(text_detail)
            # print(title)
            car_kilometer = html_detail.xpath('//div[@class="product-textbox"]/ul/li/span/text()')[2]
            #car_Kilometers.append(car_kilometer)
            car_pailiang = html_detail.xpath('//div[@class="product-textbox"]/ul/li/span/text()')[3]
            #car_Pailiangs.append(car_pailiang)
            car_gear_box = html_detail.xpath('//div[@class="product-textbox"]/ul/li/span/text()')[4]
            #car_Gearboxs.append(car_gear_box)

            name_Pinpais.append(name_Pinpai)
            year_Login_ins.append(year_Login_in)
            price_Sellers.append(price_Seller)
            price_newGuides.append(price_newGuide)
            car_Kilometers.append(car_kilometer)
            car_Pailiangs.append(car_pailiang)
            car_Gearboxs.append(car_gear_box)

            print(year_Login_in)
            print(name_Pinpai)
            print(price_Seller)
            print(price_newGuide)
            print(car_kilometer)
            print(car_pailiang)
            print(car_gear_box)
        except:
            print('代码块执行异常，该数据不要！')
    return name_Pinpais,price_Sellers,price_newGuides ,year_Login_ins ,car_Kilometers ,car_Pailiangs ,car_Gearboxs

pinpai_url = 'https://www.guazi.com/cd/dazhong/#bread'

resp_car = requests.get(pinpai_url,headers=headers)
text_car = resp_car.content.decode('utf-8')
html_car = etree.HTML(text_car)

ulname_car = html_car.xpath('//div[@class="dd-top"]/span[@class="a-box"]/a/text()')
ul_car = html_car.xpath('//div[@class="dd-top"]/span[@class="a-box"]/a/@href')
print(len(ulname_car))
print(ul_car)
car_dict = {}
detail_urls = []
name_Pinpais = []
price_Sellers = []
price_newGuides = []
#date_Password = []
year_Login_ins = []
car_Kilometers = []
car_Pailiangs = []
car_Gearboxs = []
for j in range(len(ulname_car)):
    if j >=0:
        print('====================================')
        print(f'{ulname_car[j]}品牌的车开始爬取')
        if ulname_car[j] != '大众':

            for i in range(1, 51):
                try:
                    print('====================================')
                    print(f'第{i}页开始爬取')
                    pinpai_url = 'https://www.guazi.com'+ ul_car[j].split('/#')[0] + '/o' + str(i)+'/#bread'
                   # pinpai_url = 'https://www.guazi.com/cd/dazhong/o' + str(i) + '/#bread'
                    name_Pinpais, price_Sellers, price_newGuides, year_Login_ins, car_Kilometers, car_Pailiangs, car_Gearboxs = single_pinpai(pinpai_url,headers)
                    print(pinpai_url)
                    print('====================================')
                    print(f'第{i}页结束爬取')
                    time.sleep(5)
                except:
                    print('数据异常，该页不要')
                    print('======***********************************===================')
            car_dict['车型_品牌'] = name_Pinpais
            car_dict['车主报价 '] = price_Sellers
            car_dict['新车指导价'] = price_newGuides
            car_dict['车牌注册日期'] = year_Login_ins
            car_dict['里程'] = car_Kilometers
            car_dict['排量'] = car_Pailiangs
            car_dict['变速箱'] = car_Gearboxs
            df  = pd.DataFrame(car_dict)
            df.to_csv(ulname_car[j]+'.csv')
            print('====================================')
            print(f'第{ulname_car[j]}种车结束爬取')



# for i in range(1,51):
#     pinpai_url = 'https://www.guazi.com/cd/dazhong/o'+ str(i)+'/#bread'
#     name_Pinpais, price_Sellers, price_newGuides, date_Password, year_Login_ins, car_Kilometers, car_Pailiangs, car_Gearboxs = single_pinpai(pinpai_url,headers)
# detail_urls.append(detail_url)
#     return detail_urls
#
# # 解析详情页面内容
# def parse_detail_page(url):
#     resp = requests.get(url, headers=header)
#     text = resp.content.decode('utf-8')
#     html = etree.HTML(text)
#     title = html.xpath('//div[@class="product-textbox"]/h2/text()')[0]
#     title = title.replace(r'\r\n', '').strip()
#     # print(title)
#     info = html.xpath('//div[@class="product-textbox"]/ul/li/span/text()')
#     # print(len(info))
#     infos = {}
#     # cardtime = info[0]
#     # what = [1]
#     km = info[2]
#     displacement = info[3]
#     speedbox = info[4]
#
#     infos['title'] = title
#     # infos['cardtime'] = cardtime
#     # infos['what'] = what
#     infos['km'] = km
#     infos['displacement'] = displacement
#     infos['speedbox'] = speedbox
#     return infos
#
# # 保存数据
# def save_info(infos,f):
#     f.write('{},{},{},{}\n'.format(infos['title'],infos['km'],infos['displacement'],infos['speedbox']))
# def main():
#     # 第一个url
#     base_url = 'https://www.guazi.com/cs/buy/o{}/'
#     with open('guazi_cs.csv', 'a', encoding='utf-8') as f:
#         for x in range(1,6):
#             url = base_url.format(x)
#             # 获取详情页面url
#             car_urls = get_detail_urls(url)
#             # 解析详情页面内容
#             for detail_url in car_urls:
#                 infos = parse_detail_page(detail_url)
#                 save_info(infos,f)
#
# if __name__=='__main__':
#     main()
