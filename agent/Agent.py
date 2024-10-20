import pandas as pd

from copy import deepcopy
from util.util import log_print


class Agent:

    def __init__(self, id, name, type, random_state):

        # ID must be a unique number (usually autoincremented).
        # Name is for human consumption, should be unique (often type + number).
        # Type is for machine aggregation of results, should be same for all
        # agents following the same strategy (incl. parameter settings).
        # Every agent is given a random state to use for any stochastic needs.
        # This is an np.random.RandomState object, already seeded.

        # ID必须是唯一的数字（通常是自动递增的）。
        # 名称仅供人类使用，应该是唯一的（通常是类型+数字）。
        # 类型用于机器聚合结果，对于遵循相同策略（包括参数设置）的所有代理，类型应相同。
        # 每个代理都有一个随机状态，用于任何随机需求。
        # 这是一个np.random。RandomState对象，已播种。
        self.id = id
        self.name = name
        self.type = type
        self.random_state = random_state

        if not random_state:
            raise ValueError("A valid, seeded np.random.RandomState object is required " +
                             "for every agent.Agent", self.name)
            sys.exit()

        # Kernel is supplied via kernelInitializing method of kernel lifecycle.
        # 内核是通过内核生命周期的kernelInitialization方法提供的。
        self.kernel = None

        # What time does the agent think it is?  Should be updated each time
        # the agent wakes via wakeup or receiveMessage.  (For convenience
        # of reference throughout the Agent class hierarchy, NOT THE
        # CANONICAL TIME.)

        # 代理人认为现在几点了？每次代理通过唤醒或receiveMessage唤醒时都应该更新。
        # （为了便于在整个Agent类层次结构中参考，而不是在CANONICAL时间。）
        self.currentTime = None

        # Agents may choose to maintain a log.  During simulation,
        # it should be stored as a list of dictionaries.  The expected
        # keys by default are: EventTime, EventType, Event.  Other
        # Columns may be added, but will then require specializing
        # parsing and will increase output dataframe size.  If there
        # is a non-empty log, it will be written to disk as a Dataframe
        # at kernel termination.

        # 代理可以选择维护日志。在模拟过程中，它应该存储为字典列表。
        # 默认情况下，预期的键是：EventTime、EventType、Event。
        # 可以添加其他列，但需要专门解析，并会增加输出数据帧的大小。
        # 如果有一个非空日志，它将在内核终止时作为数据帧写入磁盘。

        # It might, or might not, make sense to formalize these log Events
        # as a class, with enumerated EventTypes and so forth.
        # 将这些日志事件形式化为一个类，使用枚举的EventTypes等，可能有意义，也可能没有意义。
        self.log = []
        self.logEvent("AGENT_TYPE", type)

    ### Flow of required kernel listening methods:
    ### init -> start -> (entire simulation) -> end -> terminate

    def kernelInitializing(self, kernel):
        # Called by kernel one time when simulation first begins.
        # No other agents are guaranteed to exist at this time.

        # 在模拟首次开始时由内核调用一次。目前无法保证存在其他代理。

        # Kernel reference must be retained, as this is the only time the
        # agent can "see" it.

        # 内核引用必须保留，因为这是代理唯一可以“看到”它的时间。
        self.kernel = kernel

        log_print("{} exists!", self.name)

    def kernelStarting(self, startTime):
        # Called by kernel one time _after_ simulationInitializing.
        # All other agents are guaranteed to exist at this time.
        # startTime is the earliest time for which the agent can
        # schedule a wakeup call (or could receive a message).

        # 由内核调用一次_after_simulationInitialization。
        # 所有其他代理此时都保证存在。
        # startTime是代理可以调度唤醒呼叫（或可以接收消息）的最早时间。


        # Base Agent schedules a wakeup call for the first available timestamp.
        # Subclass agents may override this behavior as needed.

        # Base Agent为第一个可用的时间戳安排唤醒调用。子类代理可以根据需要覆盖此行为。

        log_print("Agent {} ({}) requesting kernel wakeup at time {}",
                  self.id, self.name, self.kernel.fmtTime(startTime))

        self.setWakeup(startTime)

    def kernelStopping(self):
        # Called by kernel one time _before_ simulationTerminating.
        # All other agents are guaranteed to exist at this time.

        pass

    def kernelTerminating(self):
        # Called by kernel one time when simulation terminates.
        # No other agents are guaranteed to exist at this time.

        # If this agent has been maintaining a log, convert it to a Dataframe
        # and request that the Kernel write it to disk before terminating.
        if self.log:
            dfLog = pd.DataFrame(self.log)
            dfLog.set_index('EventTime', inplace=True)
            self.writeLog(dfLog)

    ### Methods for internal use by agents (e.g. bookkeeping).

    def logEvent(self, eventType, event='', appendSummaryLog=False):
        # Adds an event to this agent's log.  The deepcopy of the Event field,
        # often an object, ensures later state changes to the object will not
        # retroactively update the logged event.

        # We can make a single copy of the object (in case it is an arbitrary
        # class instance) for both potential log targets, because we don't
        # alter logs once recorded.
        e = deepcopy(event)
        self.log.append({'EventTime': self.currentTime, 'EventType': eventType,
                         'Event'    : e})

        if appendSummaryLog: self.kernel.appendSummaryLog(self.id, eventType, e)

    ### Methods required for communication from other agents.
    ### The kernel will _not_ call these methods on its own behalf,
    ### only to pass traffic from other agents..

    def receiveMessage(self, currentTime, msg):
        # Called each time a message destined for this agent reaches
        # the front of the kernel's priority queue.  currentTime is
        # the simulation time at which the kernel is delivering this
        # message -- the agent should treat this as "now".  msg is
        # an object guaranteed to inherit from the message.Message class.

        # 每次发往此代理的消息到达内核优先级队列的前面时调用。
        # currentTime是内核传递此消息的模拟时间——代理应将其视为“现在”。
        # msg是一个保证从消息继承的对象。消息类。

        self.currentTime = currentTime

        log_print("At {}, agent {} ({}) received: {}",
                  self.kernel.fmtTime(currentTime), self.id, self.name, msg)

    def wakeup(self, currentTime):
        # Agents can request a wakeup call at a future simulation time using
        # Agent.setWakeup().  This is the method called when the wakeup time
        # arrives.

        # 代理可以在未来的模拟时间使用Agent.setWakeup（）请求唤醒调用。
        # 这是唤醒时间到达时调用的方法。

        self.currentTime = currentTime

        log_print("At {}, agent {} ({}) received wakeup.",
                  self.kernel.fmtTime(currentTime), self.id, self.name)

    ### Methods used to request services from the Kernel.  These should be used
    ### by all agents.  Kernel methods should _not_ be called directly!

    ###用于从内核请求服务的方法。这些应该被所有代理人使用。内核方法不应该直接调用！

    ### Presently the kernel expects agent IDs only, not agent references.
    ### It is possible this could change in the future.  Normal agents will
    ### not typically wish to request additional delay.

    ###目前，内核只需要代理ID，而不需要代理引用。未来这种情况可能会改变。普通代理通常不希望请求额外的延迟。
    def sendMessage(self, recipientID, msg, delay=0, tag="communication"):
        self.kernel.sendMessage(self.id, recipientID, msg, delay=delay, tag=tag)

    def setWakeup(self, requestedTime):
        self.kernel.setWakeup(self.id, requestedTime)

    def getComputationDelay(self):
        return self.kernel.getAgentComputeDelay(sender=self.id)

    def setComputationDelay(self, requestedDelay):
        self.kernel.setAgentComputeDelay(sender=self.id, requestedDelay=requestedDelay)

    def delay(self, additionalDelay):
        self.kernel.delayAgent(sender=self.id, additionalDelay=additionalDelay)

    def writeLog(self, dfLog, filename=None):
        self.kernel.writeLog(self.id, dfLog, filename)

    def updateAgentState(self, state):
        """ Agents should use this method to replace their custom state in the dictionary
            the Kernel will return to the experimental config file at the end of the
            simulation.  This is intended to be write-only, and agents should not use
            it to store information for their own later use.
        """

        self.kernel.updateAgentState(self.id, state)

    ### Internal methods that should not be modified without a very good reason.

    def __lt__(self, other):
        # Required by Python3 for this object to be placed in a priority queue.

        return ("{}".format(self.id) <
                "{}".format(other.id))
