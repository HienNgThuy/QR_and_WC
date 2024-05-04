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
        

        
        
