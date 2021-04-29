#!/bin/bash

for ARGUMENT in "$@"
do

    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)   

    case "$KEY" in
            DIFF)              DIFF=${VALUE} ;;
            VERS)    	      VERS=${VALUE} ;;     
            ITER)    	      ITER=${VALUE} ;;     
            SIM)    	      SIM=${VALUE} ;;     
            SAVE)    	      SAVE=${VALUE} ;;     
            RL)    	      RL=${VALUE} ;; 
            *)   
    esac    


done

echo "DIFF = $DIFF"
echo "VERS = $VERS"
echo "ITER = $ITER"
echo "SIM = $SIM"
echo "SAVE = $SAVE"
echo "RL = $RL"


if [ $RL == 'AC' ]
then
    server_string="167:GitHub_repos"
elif [ $RL == 'DQL' ]
then
    server_string="156:GitHub_Repositories"
fi

echo "server string = $server_string"

source ~/environments/venv_PedEnv/bin/activate

folder="ConvModel$VERS"
iteration="_$ITER"

#echo $folder
#echo $iteration

mkdir /home/rodpod21/GitHubRepositories/asynch-rl/Data/RobotEnv/${folder}
 
#rm /home/rodpod21/GitHubRepositories/asynch-rl/Data/RobotEnv/${folder}/TrainingLog.pkl
 
sshpass -p 'abcABC11!?' scp eregolin@172.30.121.${server_string}/asynch-rl/Data/RobotEnv/${folder}/\{'TrainingLog.pkl','train_params.txt','*'$iteration'*','val_history.npy','PG_training.npy'\} /home/rodpod21/GitHubRepositories/asynch-rl/Data/RobotEnv/${folder}/

python /home/rodpod21/GitHubRepositories/asynch-rl/examples/Tester_robot.py -v $VERS -i $ITER -sim $SIM -d $DIFF -s $SAVE