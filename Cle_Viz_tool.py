# Import library
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from io import BytesIO
import re

import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot

import io
import segno
from PIL import Image 
from urllib.request import urlopen

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import requests

separator_html = """
<div style="background: linear-gradient(45deg, red, orange, yellow, green, blue, indigo, violet); height: 3px;"></div>
"""

# Convert to download data as excel file
def to_excel(df: pd.DataFrame):
    in_memory_fp = BytesIO()
    df.to_excel(in_memory_fp)
    # Write the file out to disk to demonstrate that it worked.
    in_memory_fp.seek(0, 0)
    return in_memory_fp.read()

# st.title(':rainbow[Một số công cụ làm sạch dữ liệu cơ bản] :tulip::cherry_blossom::rose::hibiscus::sunflower::blossom:')

st.title(':blue[Một số công cụ làm sạch và trực quan hóa dữ liệu bằng Python]')

st.snow()

# Tạo cột bên trái cho menu
left_column = st.sidebar


tools =  ['Làm sạch dữ liệu', 'Trực quan hóa dữ liệu']

cleaning_tools = ['Nhận diện và làm sạch outliers', 'Làm sạch text', 'Lọc những giá trị Khác ngoài MA', 'Xóa cụm bất kỳ', 'Lọc và tính toán câu hỏi xếp hạng',
          'Lọc và đếm các giá trị duy nhất', 'Đếm tần suất từ/cụm từ/câu bất kỳ']

viz_tools = ['Tạo mà QR và chèn logo/hình ảnh', 'Tạo Wordcloud từ tần suất']

# Tạo menu dropdown list cho người dùng lựa chọn dự án
tool = left_column.selectbox(":blue[**Chọn loại ứng dụng muốn sử dụng:**]", tools)

# Logo 
# left_column.image('https://i.imgur.com/YbVRCS1.png')
left_column.image('./Image/SGC35_red 3.png')

# Lưu trữ chỉ số index của dự án được chọn
tool_num = tools.index(tool) + 1

# if tool_num == 1:
#     st.write('# Làm sạch dữ liệu')
    
#     cleaning_tool = left_column.selectbox(":red[**Chọn ứng dụng muốn sử dụng:**]", cleaning_tools)
#     tool_num_cle = cleaning_tools.index(cleaning_tool) + 1

#     if tool_num_cle == 1:
#         st.header('Nhận diện và làm sạch outliers')
            
#         st.write('### Chọn file áp dụng')
        
#         uploaded_file = st.file_uploader("Excel file:", accept_multiple_files=False)
        
#         if uploaded_file is not None:
#             data=pd.read_excel(uploaded_file)
#         else:
#             st.warning('Chưa chọn file!')
        
#         st.dataframe(data.head( ))
        
#         st.write('**Dữ liệu bao gồm các cột:**')
#         st.code(data.columns.tolist())
        
#         st.write('#### Chọn cột cần nhận diện outliers:')
        
#         column = st.selectbox('Tên cột', data.columns.tolist())
        
#         st.write('**Xem xét thống kê nhanh**')
        
#         st.dataframe(data[[column]].describe().applymap(lambda x: f"{x:0.2f}").T)
        
#         st.write('**Xem xét phân bố của dữ liệu của dữ liệu**')
        
#         st.write('***Boxplot***')
        
#         fig1 = px.box(data, x=column, color_discrete_sequence =['#BF3131'])

#         # ## loop through the values you want to label and add them as annotations
#         # for x in zip(["min","q1","med","q3","max"],data.quantile([0,0.25,0.5,0.75,1]).iloc[:,0].values):
#         #     fig1.add_annotation(
#         #         x=x[1],
#         #         y=0.3,
#         #         text=x[0] + ": " + str(x[1]),
#         #         showarrow=False
#         #         )
#         st.plotly_chart(fig1)
        
#         st.write('***Top 10 giá trị xuất hiện nhiều nhất***')
        
#         top_10 = data[column].value_counts().iloc[:10].rename_axis('Giá trị').reset_index()
        
#         top_10['Giá trị'] = top_10['Giá trị'].astype('category')
        
#         top_10.rename(columns = {'count':'Số lần'}, inplace = True) 
        
#         st.dataframe(top_10.T)
        
#         fig2 = px.bar(top_10, y='Số lần', x='Giá trị', text_auto='Auto', color_discrete_sequence =['#7D0A0A'])
        
#         st.plotly_chart(fig2)     
        
#         st.write('## Nhận diện outliers')   

#         st.write(''':red[**Lưu ý:**] Sau đây là 3 trong rất nhiều cách nhận diện outliers, tùy theo mục đích, loại dữ liệu và số lượng outliers sẽ áp dụng các cách thức khác nhau để làm việc với dữ liệu chứa outliers (có thể loại bỏ tất cả, loại bỏ 1 phần hoặc giữ nguyên):
#     * **Cách 1:** Nhận diện outliers dựa trên **Độ lệch chuẩn**, tính từ trung bình của dữ liệu, nếu giá trị nào cao hơn  Độ lệch chuẩn 3 lần sẽ là outliers.
#     * **Cách 2:** Nhận diện outliers dựa trên **Bách phân vị**, bất kỳ giá trị nào nằm ngoài khoảng Bách phân vị thứ 1 và 99 sẽ là outliers.
#     * **Cách 3:** Nhận diện outliers dựa trên **Tứ phân vị**, bất kỳ giá trị nào nằm ngoài khoảng Tứ phân vị sẽ là outliers.''')
#         # https://thongke.cesti.gov.vn/dich-vu-thong-ke/tai-lieu-phan-tich-thong-ke/845-thong-ke-mo-ta-trong-nghien-cuu-dai-luong-do-phan-tan
#         #region
#         #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#END#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#         #endregion
        
#         tab1, tab2, tab3 = st.tabs(["**Độ lệch chuẩn**", "**Bách phân vị**", "**Tứ phân vị**"])
        
#         def select_outliers(outliers, column):
#             st.write('Tổng số outliers:', len(outliers))
#             st.write('Phần trăm outliers:', round(len(outliers)/len(data[column])*100,2),'%')
#             st.write('Số lượng các giá trị là outliers:', outliers.nunique())
#             st.write('Các giá trị outliers:')
#             st.code(str(sorted(outliers.unique())))
#             st.write('Tần suất của mỗi outliers:')
#             fre = data[column].value_counts().loc[outliers.unique()].to_frame().sort_values(column, ascending=False)
#             fre.columns = ['Tần suất']
#             st.dataframe(fre.T)
#             # st.dataframe(data[column].value_counts().loc[outliers.unique()].to_frame().sort_index().T)
        
#         with tab1:
#             st.write('#### Độ lệch chuẩn')
            
#             ### Cách 1
#             mean = np.mean(data[column])
#             std_dev = np.std(data[column])
#             #More than 3 standard deviations from the mean an outlier
#             threshold = 3 
#             #create the condition to find outliers
#             outliers_C1 = data[column][np.abs(data[column] - mean) > threshold * std_dev]

#             select_outliers(outliers_C1, column)
        
#         with tab2:
#             st.write('#### Bách phân vị')
            
#             q_low = data[column].quantile(0.01)
#             q_hi  = data[column].quantile(0.99)
#             #create the condition to find outliers
#             outliers_C2 = data[column][(q_low >  data[column]) | (data[column] > q_hi)]

#             select_outliers(outliers_C2, column)
        
#         with tab3:
#             st.write('#### Tứ phân vị')
            
#             Q1 = data[column].quantile(0.25)
#             Q3  = data[column].quantile(0.75)

#             IQR = Q3 - Q1
#             lower = Q1 - 1.5*IQR
#             upper = Q3 + 1.5*IQR

#             #create the condition to find outliers
#             outliers_C3 = data[column][(lower >  data[column]) | (data[column] > upper)]
            
#             select_outliers(outliers_C3, column)       

#         st.write('## Làm sạch outliers')
        
#         st.write('#### Chọn phương pháp muốn áp dụng để loại bỏ outliers')
        
#         methd = st.selectbox('Phương pháp:', ["Độ lệch chuẩn", "Bách phân vị", "Tứ phân vị"])
        
#         if methd == 'Độ lệch chuẩn':
#             outliers = outliers_C1
#         elif methd == 'Bách phân vị':
#             outliers = outliers_C2
#         elif methd == 'Tứ phân vị':
#             outliers = outliers_C3  
            
#         st.write('Các outliers theo ',methd)
#         st.code(str(sorted(outliers.unique())))
#         st.write('Tổng số lượng outliers sẽ bị loại bỏ là ', len(outliers), ', chiếm ', round(len(outliers)/len(data[column])*100,2),'% dữ liệu')
        
#         df_outliers = outliers.rename_axis('Id').reset_index()
        
#         st.write('**Vị trí các outliers trong dữ liệu**')
            
#         st.dataframe(df_outliers)
            
#         # if st.button("Tải outliers và vị trí"):
#         #     file_name = 'Data_outliers_'+ column + '.xlsx'
#         #     df_outliers.to_excel(file_name, index=False, engine='xlsxwriter')
#         #     st.success("Results downloaded successfully!")
        
#         ### Download in Downloads folder
        
#         excel_data_outlier = to_excel(df_outliers)
#         file_name_outlier = 'Data_outliers_'+ column + '.xlsx'
        
#         if st.download_button(
#             "Tải outliers và vị trí",
#             excel_data_outlier,
#             file_name_outlier,
#             file_name_outlier,
#             key=file_name_outlier):
            
#             st.success("Results downloaded successfully!")
                
#         if st.button('Xóa toàn bộ outliers', type="primary"):
#             ### Drop outliers
#             data.loc[data[column].isin(outliers.unique().tolist()), column] = None
        
#             st.write('##### Dữ liệu sau khi làm sạch')
                
#             st.dataframe(data[[column]].describe().applymap(lambda x: f"{x:0.2f}").T)
        
#         # if st.button("Tải dữ liệu đã làm sạch", type="primary"):
#         #     file_name = 'Data_cleaned_'+ column + '.xlsx'
#         #     data[column].to_excel(file_name, index=False, engine='xlsxwriter')
#         #     st.success("Results downloaded successfully!")
        
#         ### Download in Downloads folder
            
#         excel_data_clean = to_excel(data[column])
#         file_name_clean = 'Data_cleaned_'+ column + '.xlsx'
        
#         if st.download_button(
#             "Tải dữ liệu đã làm sạch",
#             excel_data_clean,
#             file_name_clean,
#             file_name_clean,
#             key=file_name_clean, type="primary"):
            
#             st.success("Results downloaded successfully!")
            
#     elif tool_num_cle == 2:
        
#         st.header('Làm sạch text trong trường hợp Excel không thể nhận diện 2 đoạn text giống nhau')
            
#         st.write('### Chọn file áp dụng')
        
#         uploaded_file = st.file_uploader("Excel file:", accept_multiple_files=False)
        
#         if uploaded_file is not None:
#             data=pd.read_excel(uploaded_file)
#         else:
#             st.warning('Chưa chọn file!')
        
#         # st.write('#### Dữ liệu trước khi làm sạch:')
#         # st.dataframe(data.head( ))
        
#         for i in data.columns:
#             # Convert data type to category
#             data[i] = data[i].astype('string')
#             # Remove tab
#             data[i] = data[i].str.replace('\t', ' ')
#             # Remove end of line characters
#             data[i] = data[i].str.replace(r'[\r\n]+', ' ')
#             # Remove multiple spaces with one space
#             data[i] = data[i].str.replace('[\s]{2,}', ' ')
#             # Some lines start with a space, remove them
#             data[i] = data[i].str.replace('^[\s]{1,}', '')
#             # Some lines end with a space, remove them
#             data[i] = data[i].str.replace('[\s]{1,}$', '')
#             data[i] = data[i].str.strip()

#         st.write('#### Dữ liệu sau khi làm sạch:')
        
#         st.dataframe(data.head( ))
            
#         # if st.button("Tải dữ liệu đã làm sạch", type="primary"):
#         #     data.to_excel("Clean_text_data.xlsx", index=False, engine='xlsxwriter')
#         #     st.success("Results downloaded successfully!")
        
#         excel_data = to_excel(data)
#         file_name = "Clean_text_data.xlsx"
        
#         if st.download_button(
#             "Tải dữ liệu đã làm sạch",
#             excel_data,
#             file_name,
#             file_name,
#             key=file_name, type="primary"):
            
#             st.success("Results downloaded successfully!")
            
#     elif tool_num_cle == 3:
        
#         st.header("Lọc các giá trị 'Khác' nằm trong Multiple Answers")
            
#         st.write('### Chọn file áp dụng')
        
#         uploaded_file = st.file_uploader("Excel file:", accept_multiple_files=False)
        
#         if uploaded_file is not None:
#             data=pd.read_excel(uploaded_file)
#         else:
#             st.warning('Chưa chọn file!')
            
#         # Hàm làm sạch dữ liệu
#         def clean_text(df):
#             for i in df.columns:
#                 # Remove tab
#                 df[i] = df[i].str.replace('\t', ' ')
#                 # Remove end of line characters
#                 df[i] = df[i].str.replace(r'[\r\n]+', ' ')
#                 # Remove multiple spaces with one space
#                 df[i] = df[i].str.replace('[\s]{2,}', ' ')
#                 # Some lines start with a space, remove them
#                 df[i] = df[i].str.replace('^[\s]{1,}', '')
#                 # Some lines end with a space, remove them
#                 df[i] = df[i].str.replace('[\s]{1,}$', '')
#                 df[i] = df[i].str.strip()
        
#         clean_text(data)
#         data.fillna('', inplace=True)
            
#         st.dataframe(data.head( ))
        
#         st.write('**Dữ liệu bao gồm các cột:**')
#         st.code(data.columns.tolist())
        
#         st.write('#### Chọn cột chứa những giá trị có sẵn muốn loại bỏ:')
        
#         category = st.selectbox('Giá trị đã có:', data.columns.tolist())
        
#         category_to_remove = data[category].dropna().loc[lambda x: x != ''].tolist()
        
#         st.write('##### Các giá trị sẽ bị loại bỏ bao gồm:')
#         st.write(data[category].dropna().loc[lambda x: x != ''])
        
#         st.write('#### Chọn cột chứa dữ liệu hỗn hợp:')
        
#         column_raw = st.selectbox('Dữ liệu hỗn hợp:', data.columns.tolist())
        
#         st.dataframe(data[column_raw])
        
#         # Create a regular expression pattern to match the values to remove
#         pattern = '|'.join(map(re.escape, category_to_remove))
        
#         name = data.columns[data.columns.get_loc(column_raw)]
        
#         # if st.button("Nhấn để lọc những giá trị 'Khác'", type="primary"):
#         #region
#         #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#END#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# This Tool was created for the Customer Research Team in the Planning Department at SaigonCo.op Union of Trading Co-operatives in January 2024 to preprocessing survey data.#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# Author: reine_hi#0
#         #endregion    
#         # Use the pattern to remove the values from the DataFrame
#         data['Khác_'+name] = data[column_raw].apply(lambda x: re.sub(pattern, '', x))
        
#         st.write('##### Dữ liệu sau khi lọc:')
#         st.dataframe(data['Khác_'+name])
        
#         excel_data = to_excel(data[['Khác_'+name]])
#         file_name = 'Others_data_' + name + '.xlsx'
        
#         if st.download_button(
#             "Tải dữ liệu đã lọc",
#             excel_data,
#             file_name,
#             file_name,
#             key=file_name, type="primary"):
            
#             st.success("Results downloaded successfully!")

#     elif tool_num_cle == 4:
        
#         st.header("Xóa cụm/câu bất kỳ khi nó nằm xen lẫn các cụm/câu khác nhưng giữ lại nếu nó đứng riêng")
            
#         st.write('### Chọn file áp dụng')
        
#         uploaded_file = st.file_uploader("Excel file:", accept_multiple_files=False)
        
#         if uploaded_file is not None:
#             data=pd.read_excel(uploaded_file)
#         else:
#             st.warning('Chưa chọn file!')
            
#         def clean_text(df):
#             for i in df.columns:
#                 # Remove tab
#                 df[i] = df[i].str.replace('\t', ' ')
#                 # Remove end of line characters
#                 df[i] = df[i].str.replace(r'[\r\n]+', ' ')
#                 # Remove multiple spaces with one space
#                 df[i] = df[i].str.replace('[\s]{2,}', ' ')
#                 # Some lines start with a space, remove them
#                 df[i] = df[i].str.replace('^[\s]{1,}', '')
#                 # Some lines end with a space, remove them
#                 df[i] = df[i].str.replace('[\s]{1,}$', ' ')
#                 df[i] = df[i].str.strip()
        
#         clean_text(data)
#         data.fillna('', inplace=True)
            
#         st.dataframe(data.head( ))
        
#         st.write('**Dữ liệu bao gồm các cột:**')
#         st.code(data.columns.tolist())
        
#         st.write('#### Nhập vào cụm/câu muốn loại bỏ')
        
#         category = st.text_input('Cụm/câu')
        
#         st.write('##### Cụm/câu sẽ bị loại bỏ là:')
#         st.code(category)
        
#         st.write('#### Chọn cột chứa dữ liệu hỗn hợp:')
        
#         column_raw = st.selectbox('Dữ liệu hỗn hợp:', data.columns.tolist())
        
#         st.dataframe(data[column_raw])
        
#         name = data.columns[data.columns.get_loc(column_raw)]
        
#         # Remove the value '1. Chưa có nhu cầu;' from 'Feedback' column if it appears with other values
#         data['Mới_'+name] = data[column_raw].apply(lambda x: x.replace(category, '')
#                                                                             if x != category else x)
        
#         st.write('##### Dữ liệu sau khi lọc:')
#         st.dataframe(data['Mới_'+name])
        
#         excel_data = to_excel(data[['Mới_'+name]])
#         file_name = 'New_data_' + name + '.xlsx'
        
#         if st.download_button(
#             "Tải dữ liệu đã lọc",
#             excel_data,
#             file_name,
#             file_name,
#             key=file_name, type="primary"):
            
#             st.success("Results downloaded successfully!")
            
#     elif tool_num_cle == 5:
        
#         st.write('## Lọc và tính toán câu hỏi xếp hạng trong khảo sát')
        
#         st.write('### Chọn file áp dụng')
        
#         uploaded_file = st.file_uploader("Excel file:", accept_multiple_files=False)
        
#         if uploaded_file is not None:
#             data=pd.read_excel(uploaded_file)
#         else:
#             st.warning('Chưa chọn file!')
        
#         # Xóa những dòng NA
#         data = data.dropna(axis=0)
            
#         # Delimeter
#         delimiter = st.text_input('Các cụm/câu được phân tách bằng:', ';')
        
#         # Tách các giá trị
#         df = data.iloc[:,0].str.rsplit(delimiter, expand=True).add_prefix('A')
        
#         # Drop columns that have all cells containing less than 1 elements
#         df = df.loc[:, df.apply(lambda x: x.str.len() >= 1).any()]
        
#         # Change cells containing less than 1 element to None
#         df = df.applymap(lambda x: x if len(str(x)) >= 1 else None)
        
#         # Số lượng cột
#         num_of_col = len(df.columns)
        
#         st.code('Dữ liệu được tách thành '+str(num_of_col)+' cột')
#         st.dataframe(df)
        
#         #Tải dữ liệu đã tách
#         excel_data_split = to_excel(df)
#         file_name_split = 'Dữ liệu đã tách.xlsx'
        
#         if st.download_button(
#             "Tải dữ liệu đã tách",
#             excel_data_split,
#             file_name_split,
#             file_name_split,
#             key=file_name_split):
            
#             st.success("Results downloaded successfully!")
            
        
#         # Count các giá trị trong mỗi cột
#         df_count = df.agg({i:'value_counts' for i in df.columns})
        
#         # Tạo 1 copy
#         result = df_count.copy()
        
#         # Nhân các cột giá trị theo vị trí xếp hạng
#         for i, column in zip(reversed(range(num_of_col)), df_count.columns):
#             df_count[column] = df_count[column] * (i+1)
            
#         # Tính tổng và tỷ lệc chọn
#         df_count['Tổng'] = df_count.sum(axis=1)
#         df_count['Tỷ lệ'] = round((df_count['Tổng']/df_count['Tổng'].sum())*100,1).astype(str) + '%'
        
#         result['Tổng'] = df_count['Tổng']
#         result['Tỷ lệ'] = df_count['Tỷ lệ']
        
#         result = result.sort_values('Tổng', ascending=False)
        
#         result = result.fillna(0)
        
#         st.write('#### Số lần xuất hiện của các giá trị tại mỗi cột')
#         st.write("*Cột 'Tổng' là số lần xuất hiện nhân với xếp hạng theo cột của giá trị*")
        
#         st.dataframe(result)
        
#         # Tải kết quả
#         excel_data = to_excel(result)
#         file_name = 'Kết quả xếp hạng.xlsx'
        
#         if st.download_button(
#             "Tải kết quả xếp hạng",
#             excel_data,
#             file_name,
#             file_name,
#             key=file_name, type="primary"):
            
#             st.success("Results downloaded successfully!")     

#     elif tool_num_cle == 6:
        
#         st.write('## Lọc và đếm các giá trị duy nhất')
        
#         st.write('### Chọn file áp dụng')
        
#         uploaded_file = st.file_uploader("Excel file:", accept_multiple_files=False)
        
#         if uploaded_file is not None:
#             data=pd.read_excel(uploaded_file)
#         else:
#             st.warning('Chưa chọn file!')   
            
#         st.write('#### Dữ liệu đầu vào')
#         st.dataframe(data)
        
#         result = data.stack().value_counts().reset_index() 
#         result.columns = ['Giá trị', 'Tần suất']
        
#         st.write('#### Những giá trị duy nhất và số lượng của chúng trong dữ liệu')
#         st.dataframe(result)
        
#         # Tải kết quả
#         excel_data = to_excel(result)
#         file_name = 'Giá trị duy nhất và tần suất.xlsx'
        
#         if st.download_button(
#             "Tải kết quả lọc",
#             excel_data,
#             file_name,
#             file_name,
#             key=file_name, type="primary"):
            
#             st.success("Results downloaded successfully!")  
            
#     elif tool_num_cle == 7:
        
#         st.write('## Đếm số lần cụm từ bất kỳ xuất hiện trong dữ liệu')
        
#         st.write('### Chọn file áp dụng')
        
#         uploaded_file = st.file_uploader("Excel file:", accept_multiple_files=False)
        
#         if uploaded_file is not None:
#             data=pd.read_excel(uploaded_file)
#         else:
#             st.warning('Chưa chọn file!')   
            
#         st.write('#### Dữ liệu đầu vào')
#         st.dataframe(data)      
        
#         st.write('### Nhập cụm từ cần đếm')
#         st.write(''':red[**Lưu ý:**] Nếu có từ 2 cụm trở lên, giữa mỗi cụm cách nhau bằng dấu , - Ví dụ: Hàng hóa, Nhân viên ''')
        
#         text = st.text_input('Nhập cụm từ')
        
#         # convert input to lowercase list
#         string_list = [item.strip() for item in text.split(",")]
#         string_list_lower = [x.lower() for x in string_list]
        
#         # Join all columns in case there are multiple columns
#         data['col_join'] = data[data.columns[0:]].apply(lambda x: ','.join(x.dropna().astype(str)),axis=1)
        
#         # Hàm tìm cụm từ
#         def find_match_count(word: str, pattern: str) -> int:
#             return len(re.findall(pattern, word.lower()))
        
#         # Đếm số lần xuất hiện
#         for col in string_list_lower:
#             data[col] = data['col_join'].apply(find_match_count, pattern=col)
        
#         result = data.drop('col_join', axis=1)
        
#         st.write('#### Kết quả')
        
#         for col in string_list_lower:
#             freq = result[col].sum()
#             st.write(f'Tổng số lần xuất hiện của **{col}** trong dữ liệu là: **{freq}**')
        
#         st.dataframe(result)
        
#         # Tải kết quả
#         excel_data = to_excel(result)
#         file_name = 'Số lần cụm từ xuất hiện trong dữ liệu.xlsx'
        
#         if st.download_button(
#             "Tải kết quả",
#             excel_data,
#             file_name,
#             file_name,
#             key=file_name, type="primary"):
            
#             st.success("Results downloaded successfully!") 

if tool_num == 2:
    st.write('# Trực quan hóa dữ liệu')
    
    viz_tool = left_column.selectbox(":red[**Chọn ứng dụng muốn sử dụng:**]", viz_tools)
    tool_num_viz = viz_tools.index(viz_tool) + 1
    
    if tool_num_viz == 1:
        
        st.write('### Tạo mã QR từ đường link bất kỳ')
        
        st.write('#### Nhập đường dẫn/nội dung cần tạo QR')
        
        url = st.text_input('Đường dẫn/Nội dung:')

        st.write('#### Điều chỉnh các thông số:')

        dark = st.text_input('Màu mã QR:', 'black')
        light = st.text_input('Màu nền QR:','white')
        scale = st.number_input('Kích thước:', min_value=1, max_value=20, value=10 )

        st.write('#### Tên file muốn lưu:')
        file = st.text_input('Ví dụ: image.png:', 'QR.png')

        # Create a button to download the QR
        if st.button("Nhấn để tạo và lưu mã QR mong muốn", type="primary"):
            qr_url = segno.make(url, error='h')
            qr_url.save(file, scale=scale, dark=dark, light=light)
            st.image(file)

        st.markdown(separator_html, unsafe_allow_html=True)

        st.write('### Thêm hình ảnh logo:')
        uploaded_files = st.file_uploader("Logo/Hình ảnh:", accept_multiple_files=False)
        
        st.image(uploaded_files, width=200)

        st.write('#### Tên file muốn lưu:')
        file_logo = st.text_input('Ví dụ: image.png:', 'QR_logo.png')

        if st.button("Nhấn để tạo và lưu mã QR có logo", type="primary"):

            img = Image.open(file)
            img = img.convert('RGB')

            img_width, img_height = img.size
            
            logo_img = Image.open(uploaded_files)

            logo_max_size = img_height // 3

            logo_img.thumbnail((logo_max_size,logo_max_size),  Image.LANCZOS)

            box = ((img_width - logo_img.size[0]) // 2, (img_height - logo_img.size[1]) // 2)

            img.paste(logo_img, box)
            img.save(file_logo)
            
            st.image(file_logo)
            
    elif tool_num_viz == 2:
        
        st.write('## Tạo Wordcloud từ các từ/cụm từ/câu và tần suất của chúng')
        #https://www.kaggle.com/code/asimislam/wordcloud-basic-regex-font-colors-image-mask

        st.write('### Chọn file áp dụng')
        
        uploaded_file = st.file_uploader("Excel file:", accept_multiple_files=False)
        
        if uploaded_file is not None:
            data=pd.read_excel(uploaded_file)
        else:
            st.warning('Chưa chọn file!')   
            
        st.write('#### Dữ liệu đầu vào')
        st.dataframe(data) 
        
        st.write('#### Chọn cột tương ứng')
        
        name = st.selectbox('Chọn cột chứa từ/cụm từ/câu:', data.columns)
        fre = st.selectbox('Chọn cột chứa tần suất:', data.columns)
        
        # Generate a dictionary from the DataFrame
        word_freq = dict(zip(data[name], data[fre]))
        
        # Colormap
        colormap = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 
                    'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 
                    'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 
                    'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 
                    'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 
                    'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 
                    'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 
                    'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 
                    'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 
                    'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 
                    'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 
                    'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic',
                    'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c',
                    'tab20c_r', 'terrain', 'terrain_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r']
        #region
        #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#END#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# This Tool was created for the Customer Research Team in the Planning Department at SaigonCo.op Union of Trading Co-operatives in January 2024 to preprocessing survey data.#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# Author: reine_hi#0
        #endregion
        
        st.write('#### Điều chỉnh các thông số cho Wordcloud')
        
        wordcloud_name = st.text_input('Nhập tên file sẽ lưu:', 'wordcloud.png')
        width = st.number_input('Chiều rộng:', value=900)
        height = st.number_input('Chiều cao:', value=700)
        min_font_size = st.number_input('Kích thước chữ nhỏ nhất:', value=15)
        background_color = st.text_input('Màu nền (CSS Colors/mã HEX):', 'white')
        colormap_sel = st.selectbox('Chọn bảng màu hiển thị:', colormap)
        
        with st.expander("**Bấm để xem các bảng màu khả dụng**"):
            # st.image('https://matplotlib.org/stable/_images/sphx_glr_colormaps_001.png')
            # st.image('https://matplotlib.org/stable/_images/sphx_glr_colormaps_002.png')
            # st.image('https://matplotlib.org/stable/_images/sphx_glr_colormaps_003.png')
            # st.image('https://matplotlib.org/stable/_images/sphx_glr_colormaps_004.png')
            # st.image('https://matplotlib.org/stable/_images/sphx_glr_colormaps_005.png')
            # st.image('https://matplotlib.org/stable/_images/sphx_glr_colormaps_006.png')
            # st.image('https://matplotlib.org/stable/_images/sphx_glr_colormaps_007.png')
            st.image('./Image/sphx_glr_colormaps_001.png')
            st.image('./Image/sphx_glr_colormaps_002.png')
            st.image('./Image/sphx_glr_colormaps_003.png')
            st.image('./Image/sphx_glr_colormaps_004.png')
            st.image('./Image/sphx_glr_colormaps_005.png')
            st.image('./Image/sphx_glr_colormaps_006.png')
            st.image('./Image/sphx_glr_colormaps_007.png')
            st.write('***Nguồn:*** *https://matplotlib.org/stable/users/explain/colors/colormaps.html*')

        with st.expander("**Bấm để xem ví dụ của các bảng màu**"):
            st.image('./Image/wordcloud_colormap_sample.png')
            st.write('***Nguồn:*** *https://www.kaggle.com/code/niteshhalai/wordcloud-colormap*')
            
        # Chèn hình ảnh để điều chỉnh hình dạng wordcloud
        try:
            col1, col2 = st.columns([5,2])
            with col1:
                image_file = st.file_uploader("**Thêm hình ảnh để điều chỉnh hình dạng của Wordcloud (Có thể thêm hoặc không)**", accept_multiple_files=False)
                if image_file:
                    with col2:
                        st.write('Wordcloud có dạng:')
                        st.image(image_file, width=100)
            mask = np.array(Image.open(image_file))
        except:
            mask = None
        
        # response = requests.get("https://media.cheggcdn.com/media/216/21621ee5-e80f-47f3-9145-513f2229b390/phploeBuh.png")
        # mask = np.array(Image.open(BytesIO(response.content)))
        
        # Generate the word cloud
        wordcloud = WordCloud(width=width, height=height, background_color=background_color, min_font_size=min_font_size,
                              colormap=colormap_sel, mask=mask).generate_from_frequencies(word_freq)

        # store to file
        wordcloud.to_file(wordcloud_name)

        # Plot the word cloud
        fig = plt.figure(figsize=(8, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(fig)
        

        
        
