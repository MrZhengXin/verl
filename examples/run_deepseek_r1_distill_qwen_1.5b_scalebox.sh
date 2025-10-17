      
#!/bin/bash
# Under `primeIntellect/verifiable-coding-problems` with data selection, the model DeepSeek-R1-Distill-Qwen-1.5B's performance on LiveCodeBench(202408–202501) rises from 16.66% to 24.73% with 1520 steps
# Using the docker image of `zhengxin1999/verl-sandbox:v1` or build by `scripts/Dockerfile.verl`
# Set environment variables `RANK` and `WORLD_SIZE` 
# Login wandb

export VLLM_ATTENTION_BACKEND=XFORMERS

project_path=$HOME/verl
sandbox_path=$HOME/icip-sandbox


check_port() {
    (echo > /dev/tcp/$MASTER_ADDR/$PORT) >/dev/null 2>&1
    return $?
}

PORT=6379

#############################################################
# Start sandboxfusion service
#############################################################
current_date=$(date +"%m%d")

##### Set SERVER_DIR
export SERVER_DIR=${project_path}/server/new_multi_node_sandbox_codemath_16k_1.5B_test_${current_date}

cd $sandbox_path

# Check if the directory exists
if [ ! -d "$SERVER_DIR" ]; then
    # If the directory does not exist, create it
    mkdir -p "$SERVER_DIR"
    echo "Directory $SERVER_DIR created."
else
    # If the directory already exists, output a prompt
    echo "Directory $SERVER_DIR already exists."
fi

if [ "$WORLD_SIZE" -eq 1 ] || [ "$RANK" -ne 0 ]; then
    echo "Starting sandbox server on rank $RANK..."
    source ~/miniconda3/bin/activate
    source activate sandbox
    make run-distributed > $SERVER_DIR/sandbox_$RANK.log 2>&1 &
    conda deactivate
fi

sleep 60s

#############################################################
# Start nginx service
#############################################################
echo "currect rank is: $RANK"
if [ $RANK -eq 0 ]; then
    NUM_NODES=$(( $WORLD_SIZE - 1 ))
    MASTER_HOST=${__HOST_IP__}
    NGINX_PORT=${NGINX_PORT:-8082}

    # Set a while loop, and check if number of files $SERVER_DIR/addr_* larger than number of nodes
    while [ $(ls $SERVER_DIR/addr_* | wc -l) -lt ${NUM_NODES} ]; do
        echo "Waiting for all ${NUM_NODES} nodes to be ready..."
        sleep 5
    done

    export ADDRESS_LIST=''

    addr_list=$(ls $SERVER_DIR/addr_*)
    for addr_file in ${addr_list}; do
        # Read the address from the file
        addr=$(cat ${addr_file})
        # Check if the address is working
        if ! curl -s "http://${addr}" --max-time 2; then
            echo "Address ${addr} is not working, remove it from the list"
            rm ${addr_file}
            continue
        fi
        # Write the address to `ADDRESS_LIST`
        ADDRESS_LIST+="        server ${addr} max_fails=3 fail_timeout=30s;"
        ADDRESS_LIST+=$'\n'
        echo "Address ${addr} is working, add it to the list"
    done

    # Generate nginx config file
    cat examples/grpo_trainer/scalebox_nginx.conf.template | envsubst > $SERVER_DIR/nginx.conf

    echo "Nginx config file generated at $SERVER_DIR/nginx.conf"
    cat $SERVER_DIR/nginx.conf
    echo ""
    echo "Starting Nginx..."
    # If nginx is not already running, start it; otherwise, reload the config
    nginx_pid=$(ls /var/run/nginx.pid 2>/dev/null)
    if [ -z "${nginx_pid}" ]; then
        echo "Nginx is not running, starting it..."
        nginx -c $SERVER_DIR/nginx.conf
    else
        echo "Nginx is already running, reloading the config..."
        nginx -s reload -c $SERVER_DIR/nginx.conf
    fi
    echo "Nginx started, listening on port ${NGINX_PORT}"
    echo "You can access the server at http://localhost:${NGINX_PORT}"
fi


#############################################################
# Set the periodic output of Nginx logs
#############################################################

if [ $RANK -eq 0 ]; then
    NGINX_LOG=$SERVER_DIR/nginx.log
    # define a function to execute your task
    run_periodic_task() {
        while true; do
            {
                echo "Running at $(date)"

                echo "============= Cumulative connections ================="
                awk -F'upstream_addr=' '{print $2}' /var/log/nginx/access.log | awk '{print $1}' | sort | uniq -c | sort -rn


                addr_list=$(ls $SERVER_DIR/addr_*)
                echo "============= Active connections ================="
                # Count connections for each port
                for addr_file in $addr_list; do
                    addr=$(cat ${addr_file})
                    # Count both incoming and outgoing connections to this address
                    count=$(netstat -an | grep ESTABLISHED | grep -c "$addr ")
                    if [ $count -gt 0 ]; then
                        echo "Address $addr: $count connections"
                    else
                        echo "Address $addr: 0 connections"
                    fi
                done

                echo "============= Working servers ================="
                
                for addr_file in ${addr_list}; do
                    addr=$(cat ${addr_file})
                    if ! curl -s "http://${addr}" > /dev/null --max-time 2; then
                        echo "Address ${addr} is not working"
                        continue
                    fi
                    echo "Address ${addr} is working"
                done
            } >> "$NGINX_LOG" 2>&1

            # execute the task every 100s
            sleep 100
        done
    }

    # put the periodic task in the background
    run_periodic_task &
fi

#############################################################
# start ray
#############################################################


if [ $RANK -eq 0 ]; then
    ray start --head --port $PORT
else
    while ! check_port; do
        echo "Port $PORT on $MASTER_ADDR is not open yet. Retrying in 5 seconds..."
        sleep 30s # wait for head node to start
    done
    ray start --address=$MASTER_ADDR:$PORT
fi

echo "Ray started on rank $RANK"


#############################################################
# RL Training
#############################################################
cd ${project_path}
mkdir -p logs


current_time=$(date +"%m%d%H%M")

# export WANDB_MODE=offline 
# wandb offline

export SANDBOX_ENDPOINT='http://localhost:8082'
alias python='/usr/bin/python'

if [ $RANK -eq 0 ]; then

    mini_batch_size=256
    temperature=0.9
    clip_ratio=0.2

    max_prompt_length=$((1024 * 2))
    max_response_length=$((1024 * 8))
    max_num_batched_tokens=$((1024 * 10))
    enable_overlong_buffer=True
    overlong_buffer_len=$((1024 * 4))

    export MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    export OUTPUT_DIR="${project_path}/checkpoints/fusion_prime_1.5B_single_distill-mb32-t0.9-cr0.2-${current_time}"
    export TRAIN_FILES="[YOUR_TRAIN_FILE_PATH]"
    export VAL_FILES="[YOUR_VAL_FILE_PATH]"

    PYTHONUNBUFFERED=1 /usr/bin/python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VAL_FILES} \
    data.train_batch_size=1024 \
    data.val_batch_size=32 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_batch_size} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.clip_ratio=${clip_ratio} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    reward_model.reward_manager="prime" \
    reward_model.sandbox_fusion.url=${SANDBOX_ENDPOINT}/common_evaluate_batch \
    reward_model.sandbox_fusion.max_concurrent=64 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='code_rl' \
    trainer.experiment_name="fusion_prime_1.5B-mb${mini_batch_size}-t${temperature}-cr${clip_ratio}-${current_time}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${WORLD_SIZE} \
    trainer.save_freq=20 \
    trainer.test_freq=160 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=10 \
    trainer.default_local_dir=${OUTPUT_DIR} \
    data.filter_overlong_prompts=True \
    2>&1 | tee logs/fusion_prime_1.5B-mb${mini_batch_size}-t${temperature}-cr${clip_ratio}-${current_time}.log

    echo "Training is done on rank 0, stopping Ray..."
    ray stop --force

else
    #############################################################
    # rank != 0 processes, wait for main process to stop
    #############################################################
    echo "Worker rank $RANK is waiting for Ray to stop..."

    # (optional) if your Ray version is new, you can use ray status to detect
    while true; do
        ray status 1>/dev/null 2>&1
        if [ $? -ne 0 ]; then
            echo "Ray cluster no longer available. Exiting worker..."
            break
        fi
        sleep 5m
    done

fi

echo "Rank $RANK script ended."
