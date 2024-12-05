import tkinter as tk
from datetime import datetime

def update_time():
    # 获取当前时间
    current_time = datetime.now()
    
    # ISO 8601 格式时间
    iso_time = current_time.isoformat()
    
    # 转换为常见的时间格式：年-月-日 时:分:秒.毫秒
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # 显示到毫秒
    
    # 获取Unix时间戳（毫秒）
    unix_timestamp = int(current_time.timestamp() * 1000)
    
    # 更新标签内容，显示 ISO 时间、常见时间格式和 Unix 时间戳
    time_label.config(text=f"ISO timestamp: {iso_time}\n"
                           f"Converted time: {formatted_time}\n"
                           f"Unix timestamp (ms): {unix_timestamp}")
    
    # 设定0.01秒后再调用自己，从而每秒更新100次
    root.after(10, update_time)

# 创建主窗口
root = tk.Tk()
root.title("timestamp real-time display")

# 创建显示时间的标签
time_label = tk.Label(root, text="", font=("Helvetica", 16))
time_label.pack(padx=20, pady=20)

# 开始更新时间
update_time()

# 运行主循环
root.mainloop()
