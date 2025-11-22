# --------- LẤY SẢN PHẨM ĐÃ TƯƠNG TÁC ---------
purchased_ids = [] #Khởi tạo danh sách rỗng để chứa product_id người dùng đã mua
if not data['purchases'].empty: #Kiểm tra nếu data mua hàng k rỗng 
    purchased_ids = data['purchases'][data['purchases']["user_id"] == user_id]["product_id"].unique() #Lấy danh sách sản phẩm người dùng đã mua

browsed_ids = []   #Khởi tạo danh sách rỗng để chứa product_id người dùng đã xem 
if not data['browsing_history'].empty: #Kiểm tra nếu data lịch sử xem k rỗng  
    browsed_ids = data['browsing_history'][data['browsing_history']["user_id"] == user_id]["product_id"].unique()  #Lấy danh sách sản phẩm người dùng đã xem

interacted_ids = np.union1d(purchased_ids, browsed_ids)  #Gộp các sản phẩm đã tương tác 
interacted_products = data['products'][data['products']["product_id"].isin(interacted_ids)].head(8) #Lấy tối đa 8 sản phẩm trong bảng products mà đã tương tác

if not interacted_products.empty:  #Kiểm tra nếu sản phẩm tương tác k rỗng 
    render_product_grid(interacted_products, data['product_images'], "Sản phẩm đã xem", user_id, "interacted")  #Hiển thị sản phẩm đã tương tác dưới dạng lưới (grid) kèm hình ảnh

st.subheader("Sản phẩm đã tương tác") #Hiển thị tiêu đề 
st.dataframe(interacted)              #Hiển thị bảng tương tác

# --------- CHẠY THUẬT TOÁN GỢI Ý ---------
def get_recommendations(user_id: int, algorithm: str, data: dict):
    try:              #Bắt đầu khối xử lí lỗi 
        if algorithm == "collaborative":
            from model import collaborative_filtering
            return collaborative_filtering(user_id, data['purchases'], data['products'])
        #Kiểm tra nếu là collaborative, gọi hàm collaborative_filtering để tạo gợi ý dựa trên hành vi người dùng khác
        elif algorithm == "content-based":
            from model import content_based_filtering
            return content_based_filtering(
                user_id, data['purchases'], data['browsing_history'], data['products']
            )
        #Nếu là content-based, gọi hàm content_based_filtering dựa trên nội dung sản phẩm
        elif algorithm == "hybrid":
            from model import hybrid_recommendation
            return hybrid_recommendation(
                user_id, data['purchases'], data['browsing_history'], data['products']
            )
        #Nếu là hybrid, gọi hàm hybrid_recommmendation dựa trên hợp 2 thuật toán collaborative và content-based
        elif algorithm == "multi-modal":   #Kiểm tra thuật toán multi-modal
            try:
                import torch
                from model import MultiModalModel
                ...
                sample_products["score"] = np.random.random(len(sample_products))  #Gán score ngẫu nhiên cho sản phẩm
                sample_products["source"] = "Multi-Modal"                          #Gán source là Multi-Modal
                return sample_products
            except ImportError:     #Nếu có lỗi xảy ra 
                st.warning("Thiếu thư viện torch để chạy multi-modal.")  #Hiển thị cảnh báo 
                return pd.DataFrame()    #Trả về dataframe rỗng 
        else:  #Nếu thuật toán k trùng với case nào, trả về dataframe rỗng 
            return pd.DataFrame()
    except Exception as e:   #Nếu lỗi xảy ra ở khối try, bắt exception và xử lý
        st.error(f"Lỗi khi tạo gợi ý: {e}")  #Hiển thị thông báo lỗi 
        return pd.DataFrame()                #Trả về dataframe rỗng 
    
# --------- HẬU XỬ LÝ VÀ HIỂN THỊ KẾT QUẢ ---------
if not recommendations.empty:  #Kiểm tra nếu gợi ý k rỗng 
    new_recs = recommendations[~recommendations["product_id"].isin(interacted_ids)].head(8) #Lọc ra những sản phẩm đã tương tác
    if not new_recs.empty:    #Nếu còn gợi ý mới 
        render_product_grid(new_recs, data['product_images'], "Gợi ý cho bạn", user_id, "recommendations") #Hiển thị sản phẩm gợi ý cho người dùng dưới dạng lưới (grid) kèm hình ảnh
    else: #Nếu k có gợi ý mới  
        st.info("Không có gợi ý mới nào.") #Hiển thị thông báo 
else:  #Nếu gợi ý rỗng 
    st.info("Không có gợi ý khả dụng.") #Hiển thị thông báo
