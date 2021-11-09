from sys import exit
from bitcoin.core.script import *

from utils import *
from config import my_private_key, my_public_key, my_address, faucet_address
from ex1 import send_from_P2PKH_transaction


######################################################################
# TODO: Complete the scriptPubKey implementation for Exercise 3
ex3a_txout_scriptPubKey = [OP_2DUP,OP_ADD,190,OP_EQUALVERIFY,OP_SUB,1440,OP_EQUAL]
'''
通过线性方程组的解（x，y）赎回结果
因为要验证x+y 和x-y，因此需要两份x，y  ；所以首先使用OP_2DUP复制栈顶的两个元素，也就是(x,y)
然后开始依次验证x+y是否为190  x-y是否为1440
首先使用OP_ADD，弹出栈顶前两个元素，也就是我们复制的x和y，进行相加，然后将结果压入栈中
然后我们将190 也就是我们学号的第一部分，用来判断刚刚压入栈内的结果是否等于190
然后使用OP_EQUALVERIFY 验证x+y==190？ 如果等于则继续运行脚本，不等于则停止运行；
不用OP_EQUAL是因为他会产生一个返回值，这样的话我们需要额外把返回值弹出，这会使得脚本变长
到这里，堆栈里面就只剩下x和y了，类似于add，使用OP_SUB将x和y弹出栈进行减法操作，并把结果压入栈
我们把要验证的学号的第二部分1440放入栈，使用OP_EQUAL对二者进行比较，这里用OP_EQUAL而不用OP_EQUALVERIFY的原因是因为
OP_EQUALVERIFY本质上是先执行OP_EQUAL，如果结果为0再执行OP_VERIFY，因此为了尽可能的使我们的脚本更加简洁，因此使用OP_EQUAL
'''
######################################################################

if __name__ == '__main__':
    ######################################################################
    # TODO: set these parameters correctly
    amount_to_send = 0.00008
    txid_to_spend = (
        '394177e9c2e971c99b20781c2d970a5ff6a482ef871216746b9db4e350967558') #当初的分币交易id
    utxo_index = 2  #选择分币交易的第三笔输出进行花销
    ######################################################################

    response = send_from_P2PKH_transaction(
        amount_to_send, txid_to_spend, utxo_index,
        ex3a_txout_scriptPubKey)
    print(response.status_code, response.reason)
    print(response.text)
