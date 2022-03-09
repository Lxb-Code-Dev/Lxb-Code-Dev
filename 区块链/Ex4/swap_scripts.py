from bitcoin.core.script import *

######################################################################
# This function will be used by Alice and Bob to send their respective
# coins to a utxo that is redeemable either of two cases:
# 1) Recipient provides x such that hash(x) = hash of secret 
#    and recipient signs the transaction.
# 2) Sender and recipient both sign transaction
# 
# TODO: Fill this in to create a script that is redeemable by both
#       of the above conditions.
# 
# See this page for opcode: https://en.bitcoin.it/wiki/Script
#
#

# This is the ScriptPubKey for the swap transaction
def coinExchangeScript(public_key_sender, public_key_recipient, hash_of_secret):
    return [
        OP_DEPTH, 2, OP_EQUAL,   #通过OP_DEPTH判断栈大小，因为解锁脚本长度不一样，因此可以通过栈大小判断是哪个解锁脚本
        OP_IF,                   #如果OP_EQUAL判断的结果为true则执行OP_IF后的分支，为false则执行OP_ELSE分支
        OP_HASH160, hash_of_secret, OP_EQUALVERIFY, public_key_recipient, OP_CHECKSIG, OP_ELSE,  #想要解锁需要正确的secret以及自己签名，
                                                                        # 开锁后，另一方能够在链上看到secret，可以根据secret和自己的签名解锁对方的转账交易
        2, public_key_sender, public_key_recipient, 2, OP_CHECKMULTISIG,               # 或者是根据自己和对方的签名进行解锁，参考实验Ex2
        OP_ENDIF
    ]

# This is the ScriptSig that the receiver will use to redeem coins
def coinExchangeScriptSig1(sig_recipient, secret):
    return [sig_recipient,secret]

# This is the ScriptSig for sending coins back to the sender if unredeemed
def coinExchangeScriptSig2(sig_sender, sig_recipient):
    return [OP_0, sig_sender, sig_recipient] #跟实验Ex2一样，加OP_0是由于设计时的缺陷

#
#
######################################################################

