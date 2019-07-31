# 如何给服务器绑定公网 ip

- [总的来说还不错的教程](https://www.zybuluo.com/Tyhj/note/1018828)
- [通过 curl ifconfig.me 得到的公网ip 不一定行，上一篇文章中也提到了这个问题](https://www.zhihu.com/question/67131046)
- [这里面介绍了一大堆方法，但不是直接利用公网ip那一套](https://www.zhihu.com/question/27771692)
  - “买个最便宜的VPS，然后在家里的Linux上用ssh reverse channel连过去就行了。”
  - ngrok 解决方案
- [这里主要介绍了如何检查 sshd 服务是否开启](https://linux.it.net.cn/e/server/ssh/2015/0501/14838.html)
  - sudo iptables -A INPUT -p tcp --dport ssh -j ACCEPT 这个命令可以开启 iptable 权限

## current state

- sshd 服务确认开启
- iptables 确认允许
- 通过 curl ifconfig.me 获得了一个公网 ip
- 但是 connection refused
  - 且 telnet IP PORT 不通

## 最终的解决方案

- [ngrok](https://www.ngrok.cc/)

背后的原理貌似和 ssh reverse tunnel 是一样的，就不做深究了

但是要注意这种方法需要把通道映射到 22 端口，也就是 ssh 的 tunnel 端口，不能随便映射一个