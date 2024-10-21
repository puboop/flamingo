import numpy as np
import sys


class LatencyModel:
    """
    LatencyModel provides a latency model for messages in the ABIDES simulation.  The default
    is a cubic model as described herein.

    latency model为ABIDES模拟中的消息提供了一个延迟模型。默认值是本文所述的立方体模型。

    Model parameters may either be passed as kwargs or a single dictionary with a key named 'kwargs'.
    模型参数可以作为kwargs传递，也可以通过一个名为“kwargs”的关键字传递。

    Using the 'cubic' model, the final latency for a message is computed as: min_latency + [ a / (x^3) ],
    where 'x' is randomly drawn from a uniform distribution (jitter_clip,1], and 'a' is the jitter
    parameter defined below.
    使用“立方体”模型，消息的最终延迟计算为：min_latency+[a/（x^3）]，其中“x”是从均匀分布（jitter_clip，1]中随机抽取的，“a”是下面定义的抖动参数。

    The 'cubic' model requires five parameters (there are defaults for four).  Scalar values
    apply to all messages between all agents.  Numpy array parameters are all indexed by simulation
    agent_id.  Vector arrays (1-D) are indexed to the sending agent.  For 2-D arrays of directional
    pairwise values, row index is the sending agent and column index is the receiving agent.
    These do not have to be symmetric.
    “立方体”模型需要五个参数（默认值为四个）。标量值适用于所有代理之间的所有消息。Numpy数组参数都由模拟代理_id索引。向量数组（1-D）被索引到发送代理。
    对于双向成对值的二维数组，行索引是发送代理，列索引是接收代理。这些不一定是对称的。

    'connected' must be either scalar True or a 2-D numpy array.  A False array entry prohibits
    communication regardless of values in other parameters.  Boolean.  Default is scalar True.
    “connected”必须是标量True或二维numpy数组。False数组条目禁止通信，而不管其他参数中的值如何。布尔值。默认值为标量True。

    'min_latency' requires a 2-D numpy array of pairwise minimum latency.  Integer nanoseconds.
    No default value.
    “min_latency”需要成对最小延迟的二维numpy数组。整数纳秒。无默认值。

    'jitter' requires a scalar, a 1-D numpy vector, or a 2-D numpy array.  Controls shape of cubic
    curve for per-message additive latency noise.  This is the 'a' parameter in the cubic equation above.
    Float in range [0,1].  Default is scalar 0.5.
    “抖动”需要标量、1-D numpy向量或2-D numpy数组。控制每条消息附加延迟噪声的三次曲线形状。
    这是上述三次方程中的“a”参数。浮动范围[0,1]。默认值为标量0.5。

    'jitter_clip' requires a scalar, a 1-D numpy vector, or a 2-D numpy array.  Controls the minimum value
    of the uniform range from which 'x' is selected when applying per-message noise.  Higher values
    create a LOWER maximum value for latency noise (clipping the cubic curve).  Parameter is exclusive:
    'x' is drawn from (jitter_clip,1].  Float in range [0,1].  Default is scalar 0.1.
    “jitter_clip”需要标量、1-D numpy向量或2-D numpy数组。控制应用每条消息噪声时选择“x”的均匀范围的最小值。
    较高的值会为延迟噪声创建较低的最大值（剪切三次曲线）。
    参数是独占的：“x”从（jitter_clip，1]中提取。浮点数在[0,1]范围内。默认值为标量0.1。

    'jitter_unit' requires a scalar, a 1-D numpy vector, or a 2-D numpy array.  This is the fraction of
    min_latency that will be considered the unit of measurement for jitter.  For example,
    if this parameter is 10, an agent pair with min_latency of 333ns will have a 33.3ns unit of measurement
    for jitter, and an agent pair with min_latency of 13ms will have a 1.3ms unit of measurement for jitter.
    Assuming 'jitter' = 0.5 and 'jitter_clip' = 0, the first agent pair will have 50th percentile (median)
    jitter of 133.3ns and 90th percentile jitter of 16.65us, and the second agent pair will have 50th percentile
    (median) jitter of 5.2ms and 90th percentile jitter of 650ms.  Float.  Default is scalar 10.
    “jitter_unit”需要标量、1-D numpy向量或2-D numpy数组。这是min_latency的分数，
    将被视为抖动的测量单位。例如，如果此参数为10，则最小延迟为333ns的代理对将具有33.3ns的抖动测量单位，
    最小延迟为13ms的代理对则将具有1.3ms的抖动测量单元。假设“抖动”=0.5，“jitter_clip”=0，
    则第一个代理对的第50百分位（中值）抖动为133.3ns，第90百分位抖动为16.65us，
    第二个代理对将具有5.2ms的第50百分位（中位数）抖动和650ms的第90百份位抖动。漂浮。默认值为标量10。
    1秒（s）等于1000毫秒（ms）
    1毫秒（ms）等于1000微秒（μs）
    1微秒（μs）等于1000纳秒（ns）

    All values except min_latency may be specified as a single scalar for simplicity, and have defaults to
    allow ease of use as: latency = LatencyModel('cubic', min_latency = some_array).
    为了简单起见，除min_latency之外的所有值都可以指定为单个标量，并且具有默认值，
    以便于使用：latency=latency Model（'cubic'，min_latency=some_array）。

    All values may be specified with directional pairwise granularity to permit quite complex network models,
    varying quality of service, or asymmetric capabilities when these are necessary.
    所有值都可以用定向成对粒度指定，以允许非常复杂的网络模型、不同的服务质量或必要时的不对称能力。

    Selection within the range is from a cubic distribution, so extreme high values will be
    quite rare.  The table below shows example values based on the jitter parameter a (column
    header) and x drawn from a uniform distribution from [0,1] (row header).
    该范围内的选择来自三次分布，因此极高的值将非常罕见。下表显示了基于抖动参数a（列标题）和x的示例值，这些值来自[0,1]（行标题）的均匀分布。

        x \ a	0.001	0.10	0.20	0.30	0.40	0.50	0.60	0.70	0.80	0.90	1.00
        0.001	1M	100M	200M	300M	400M	500M	600M	700M	800M	900M	1B
        0.01	1K	100K	200K	300K	400K	500K	600K	700K	800K	900K	1M
        0.05	8.00	800.00	1.6K	2.4K	3.2K	4.0K	4.8K	5.6K	6.4K	7.2K	8.0K
        0.10	1.00	100.00	200.00	300.00	400.00	500.00	600.00	700.00	800.00	900.00	1,000.00
        0.20	0.13	12.50	25.00	37.50	50.00	62.50	75.00	87.50	100.00	112.50	125.00
        0.30	0.04	3.70	7.41	11.11	14.81	18.52	22.22	25.93	29.63	33.33	37.04
        0.40	0.02	1.56	3.13	4.69	6.25	7.81	9.38	10.94	12.50	14.06	15.63
        0.50	0.01	0.80	1.60	2.40	3.20	4.00	4.80	5.60	6.40	7.20	8.00
        0.60	0.00	0.46	0.93	1.39	1.85	2.31	2.78	3.24	3.70	4.17	4.63
        0.70	0.00	0.29	0.58	0.87	1.17	1.46	1.75	2.04	2.33	2.62	2.92
        0.80	0.00	0.20	0.39	0.59	0.78	0.98	1.17	1.37	1.56	1.76	1.95
        0.90	0.00	0.14	0.27	0.41	0.55	0.69	0.82	0.96	1.10	1.23	1.37
        0.95	0.00	0.12	0.23	0.35	0.47	0.58	0.70	0.82	0.93	1.05	1.17
        0.99	0.00	0.10	0.21	0.31	0.41	0.52	0.62	0.72	0.82	0.93	1.03
        1.00	0.00	0.10	0.20	0.30	0.40	0.50	0.60	0.70	0.80	0.90	1.00
    """

    def __init__(self, latency_model='cubic', random_state=None, **kwargs):
        """
        Model-specific parameters may be specified as keyword args or a dictionary with key 'kwargs'.

        Required keyword parameters:
          'latency_model' : 'cubic'

        Optional keyword parameters:
          'random_state'  : an initialized np.random.RandomState object.
        """

        self.latency_model = latency_model.lower()
        self.random_state = random_state

        # This permits either keyword args or a dictionary of kwargs.  The two cannot be mixed.
        if 'kwargs' in kwargs: kwargs = kwargs['kwargs']

        # Check required parameters and apply defaults for the selected model.
        if (latency_model.lower() == 'cubic'):
            if 'min_latency' not in kwargs:
                print("Config error: cubic latency model requires parameter 'min_latency' as 2-D ndarray.")
                sys.exit()

            # Set defaults.
            kwargs.setdefault('connected', True)
            kwargs.setdefault('jitter', 0.5)
            kwargs.setdefault('jitter_clip', 0.1)
            kwargs.setdefault('jitter_unit', 10.0)
        elif (latency_model.lower() == 'deterministic'):
            if 'min_latency' not in kwargs:
                print("Config error: deterministic latency model requires parameter 'min_latency' as 2-D ndarray.")
                sys.exit()
        else:
            print(f"Config error: unknown latency model requested ({latency_model.lower()})")
            sys.exit()

        # Remember the kwargs for use generating jitter (latency noise).
        self.kwargs = kwargs

    def get_latency(self, sender_id=None, recipient_id=None):
        """
        LatencyModel.get_latency() samples and returns the final latency for a single Message according to the
        model specified during initialization.

        Required parameters:
          'sender_id'    : simulation agent_id for the agent sending the message
          'recipient_id' : simulation agent_id for the agent receiving the message
        """

        kw = self.kwargs
        # 得到具体的一个值
        min_latency = self._extract(kw['min_latency'], sender_id, recipient_id)

        if self.latency_model == 'cubic':
            # Generate latency for a single message using the cubic model.

            # If agents cannot communicate in this direction, return special latency -1.
            if not self._extract(kw['connected'], sender_id, recipient_id): return -1

            # Extract the cubic parameters and compute the final latency.
            # 提取三次参数并计算最终延迟。
            a = self._extract(kw['jitter'], sender_id, recipient_id)
            clip = self._extract(kw['jitter_clip'], sender_id, recipient_id)
            unit = self._extract(kw['jitter_unit'], sender_id, recipient_id)
            # Jitter requires a uniform random draw.
            # 抖动需要均匀的随机抽取。
            x = self.random_state.uniform(low=clip, high=1.0)

            # Now apply the cubic model to compute jitter and the final message latency.
            # 现在应用三次模型来计算抖动和最终消息延迟。
            latency = min_latency + ((a / x ** 3) * (min_latency / unit))

        elif self.latency_model == 'deterministic':
            return min_latency

        return latency

    def _extract(self, param, sid, rid):
        """
        Internal function to extract correct values for a sender->recipient pair from parameters that can
        be specified as scalar, 1-D ndarray, or 2-D ndarray.
        内部函数，用于从可以指定为标量、1-D ndarray或2-D ndarray的参数中提取发送方->接收方对的正确值。

        Required parameters:
          'param' : the parameter (not parameter name) from which to extract a value
          'sid'   : the simulation sender agent id
          'rid'   : the simulation recipient agent id
        """
        # 判断是否为标量
        if np.isscalar(param): return param

        if type(param) is np.ndarray:
            if param.ndim == 1:
                return param[sid]
            elif param.ndim == 2:
                # 索引到具体的值
                return param[sid, rid]

        print("Config error: LatencyModel parameter is not scalar, 1-D ndarray, or 2-D ndarray.")
        sys.exit()
