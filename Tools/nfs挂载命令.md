## NFS 挂载命令

1. 安装 nfs-common

   ```shell
   apt install nfs-common
   ```

2. 创建文件夹

   ```shell
   mkdir -p /root/alex-data
   ```

3. 执行挂载

   ```shell
   mount -t nfs x.x.x.x:/mnt/SimpleShare/sample-data/files/alex-data /root/alex-data
   ```

   
