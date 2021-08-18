#!/bin/bash

proc_name="./scripts/deploy.sh 0 rerank-dual-bert-gray-writer" #进程名字
proc_num()
{
    num=`ps -ef | grep $proc_name | grep -v grep | wc -l`
    return $num
}
proc_num
number=$?
echo $number
if [ $number -eq 0 ]
then
    cd /home/johntianlan/sources/MyReDial/myredial
    nohup ./scripts/deploy.sh 0 rerank-dual-bert-gray-writer &
fi

proc_name="./scripts/deploy.sh 1 gpt2lm" #进程名字
proc_num()
{
    num=`ps -ef | grep $proc_name | grep -v grep | wc -l`
    return $num
}
proc_num
number=$?
echo $number
if [ $number -eq 0 ]
then
    cd /home/johntianlan/sources/MyReDial/myredial
    nohup ./scripts/deploy.sh 1 rerank-gpt2lm &
fi
