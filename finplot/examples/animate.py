#!/usr/bin/env python3

# 导入所需的库
import finplot as fplt
import numpy as np
import pandas as pd

# 定义动画的帧率
FPS = 30
# 初始化动画计数器
anim_counter = 0


def gen_dumb_price():
    """
    生成模拟价格数据。

    返回:
    - df: 包含模拟价格和成交量数据的Pandas DataFrame。
    """
    # 开始生成四个随机列
    v = np.random.normal(size=(1000, 4))
    df = pd.DataFrame(v, columns='low open close high'.split())
    # 平滑处理
    df = df.rolling(10).mean()
    # 添加一点推动
    ma = df['low'].rolling(20).mean().diff()
    for col in df.columns:
        df[col] *= ma * 100
    # 使low为最低，high为最高，open和close在中间
    df.values.sort(axis=1)
    # 添加随时间变化的价格波动和一些振幅
    df = (df.T + np.sin(df.index / 87) * 3 + np.cos(df.index / 201) * 5).T + 20
    # 生成一些绿色和红色的蜡烛图
    flip = df['open'].shift(-1) <= df['open']
    df.loc[flip, 'open'], df.loc[flip, 'close'] = df['close'].copy(), df['open'].copy()
    # 价格行为导致成交量
    df['volume'] = df['high'] - df['low']
    # 设置时间戳
    df.index = np.linspace(1608332400 - 60 * 1000, 1608332400, 1000)
    return df['open close high low volume'.split()].dropna()


def gen_spots(ax, df):
    """
    生成图表上的点。

    参数:
    - ax: 绘图轴。
    - df: 数据框架。
    """
    spot_ser = df['low'] - 0.1
    spot_ser[(spot_ser.reset_index(drop=True).index - anim_counter) % 20 != 0] = np.nan
    spot_plot.plot(spot_ser, style='o', color=2, width=2, ax=ax, zoomscale=False)


def gen_labels(ax, df):
    """
    生成图表上的标签。

    参数:
    - ax: 绘图轴。
    - df: 数据框架。
    """
    y_ser = df['volume'] - 0.1
    y_ser[(y_ser.reset_index(drop=True).index + anim_counter) % 50 != 0] = np.nan
    dft = y_ser.to_frame()
    dft.columns = ['y']
    dft['text'] = dft['y'].apply(lambda v: str(round(v, 1)) if v > 0 else '')
    label_plot.labels(dft, ax=ax)


def move_view(ax, df):
    """
    移动视图。

    参数:
    - ax: 绘图轴。
    - df: 数据框架。
    """
    global anim_counter
    x = -np.cos(anim_counter / 100) * (len(df) / 2 - 50) + len(df) / 2
    w = np.sin(anim_counter / 100) ** 4 * 50 + 50
    fplt.set_x_pos(df.index[int(x - w)], df.index[int(x + w)], ax=ax)
    anim_counter += 1


def animate(ax, ax2, df):
    """
    动画函数，结合生成点、标签和移动视图的功能。

    参数:
    - ax: 第一个绘图轴。
    - ax2: 第二个绘图轴。
    - df: 数据框架。
    """
    gen_spots(ax, df)
    gen_labels(ax2, df)
    move_view(ax, df)


# 生成模拟价格数据
df = gen_dumb_price()
# 创建图表
ax, ax2 = fplt.create_plot('Things move', rows=2, init_zoom_periods=100, maximize=False)
# 绘制蜡烛图
df.plot(kind='candle', ax=ax)
# 绘制成交量图
df[['open', 'close', 'volume']].plot(kind='volume', ax=ax2)
# 初始化动态点和标签
spot_plot, label_plot = fplt.live(2)
# 设置定时器回调，以指定的帧率更新动画
fplt.timer_callback(lambda: animate(ax, ax2, df), 1 / FPS)
# 显示图表
fplt.show()
