best_eeil_configs = {
    # 0.3816 on test set
    1 : { 
        "epochs" : 70,
        "lr" : 0.1,
        "mem_size" : 3000,
        "sched_epochs" : 30,
        "ft_epochs" : 30,
        "optim" : "sgd",
        "momentum" : 0.9,
        "temp" : 3
    }, 
    #0.3709
    2 : { 
        "epochs" : 100,
        "lr" : 0.1,
        "mem_size" : 3000,
        "sched_epochs" : 30,
        "ft_epochs" : 30,
        "optim" : "sgd",
        "momentum" : 0.5,
        "temp" : 3
    }, 
    #0.3703
    3 : { 
        "epochs" : 90,
        "lr" : 0.1,
        "mem_size" : 3000,
        "sched_epochs" : 30,
        "ft_epochs" : 30,
        "optim" : "sgd",
        "momentum" : 0.5,
        "temp" : 3
    }, 
    #0.3703
    4 : { 
        "epochs" : 90,
        "lr" : 0.1,
        "mem_size" : 3000,
        "sched_epochs" : 30,
        "ft_epochs" : 30,
        "optim" : "sgd",
        "momentum" : 0.5,
        "temp" : 2
    }, 
    #0.3703
    5 : { 
        "epochs" : 90,
        "lr" : 0.1,
        "mem_size" : 3000,
        "sched_epochs" : 30,
        "ft_epochs" : 30,
        "optim" : "sgd",
        "momentum" : 0.5,
        "temp" : 2
    },
    #0.3691
    6 : { 
        "epochs" : 140,
        "lr" : 0.01,
        "mem_size" : 3000,
        "sched_epochs" : 30,
        "ft_epochs" : 30,
        "optim" : "sgd",
        "momentum" : 0.9,
        "temp" : 3
    },
    #0.3673
    7 : { 
        "epochs" : 100,
        "lr" : 0.1,
        "mem_size" : 3000,
        "sched_epochs" : 30,
        "ft_epochs" : 30,
        "optim" : "sgd",
        "momentum" : 0.5,
        "temp" : 2
    }, 
    #0.3612
    8 : { 
        "epochs" : 100,
        "lr" : 0.1,
        "mem_size" : 3000,
        "sched_epochs" : 30,
        "ft_epochs" : 30,
        "optim" : "sgd",
        "momentum" : 0.9,
        "temp" : 3
    },
    #0.3607
    9 : { 
        "epochs" : 70,
        "lr" : 0.01,
        "mem_size" : 3000,
        "sched_epochs" : 30,
        "ft_epochs" : 30,
        "optim" : "sgd",
        "momentum" : 0.9,
        "temp" : 3
    },
    #0.3583
    10 : { 
        "epochs" : 90,
        "lr" : 0.01,
        "mem_size" : 3000,
        "sched_epochs" : 30,
        "ft_epochs" : 30,
        "optim" : "sgd",
        "momentum" : 0.9,
        "temp" : 3
    },

    
}

best_lwf_res32_configs = {
    #0.3289
    1 : "epochs=50_lr=0.01_temp=2_optim=adam_lr_sched=30_alpha=1.5_early_stop=False",

    #0.327
    2 : "epochs=50_lr=0.01_temp=2_optim=adam_lr_sched=0_alpha=1_early_stop=True",

    #0.3213
    3: "epochs=50_lr=0.01_temp=2_optim=adam_lr_sched=0_alpha=2_early_stop=True",

    #0.3197
    4: "epochs=50_lr=0.01_temp=2_optim=adam_lr_sched=30_alpha=1.5_early_stop=True",

    #0.3186
    5: "epochs=90_lr=0.01_temp=2_optim=adam_lr_sched=0_alpha=1_early_stop=False",

    #0.3162
    6: "epochs=50_lr=0.01_temp=2_optim=adam_lr_sched=0_alpha=1.5_early_stop=False",

    #0.3144
    7: "epochs=50_lr=0.01_temp=2_optim=adam_lr_sched=0_alpha=1_early_stop=False",

    #0.3131
    8: "epochs=90_lr=0.01_temp=2_optim=adam_lr_sched=0_alpha=1_early_stop=True",

    #0.3123
    9: "epochs=90_lr=0.01_temp=2_optim=adam_lr_sched=0_alpha=1.5_early_stop=False",

    #0.3112
    10: "epochs=90_lr=0.01_temp=2_optim=adam_lr_sched=0_alpha=2_early_stop=False"
}


best_lwf_slimres18_configs = {
    #0.3665
    1 : "epochs=50_lr=0.01_temp=2_optim=adam_lr_sched=30_alpha=1_early_stop=True",

    #0.3359
    2 : "epochs=50_lr=0.01_temp=2_optim=adam_lr_sched=0_alpha=1_early_stop=True	",

    #0.3356
    3 : "epochs=120_lr=0.01_temp=2_optim=adam_lr_sched=40_alpha=1_early_stop=True",

    #0.3285
    4 : "epochs=50_lr=0.01_temp=2_optim=adam_lr_sched=0_alpha=1.5_early_stop=False",

    #0.3254
    5 : "epochs=70_lr=0.01_temp=2_optim=adam_lr_sched=30_alpha=2_early_stop=True",

    #0.3234
    6 : "epochs=70_lr=0.01_temp=2_optim=adam_lr_sched=30_alpha=1_early_stop=True",
    
    #0.3246
    7: "epochs=50_lr=0.01_temp=2_optim=adam_lr_sched=40_alpha=1.5_early_stop=True",

    #0.3221
    8 : "epochs=90_lr=0.01_temp=2_optim=adam_lr_sched=30_alpha=1_early_stop=False",

    #0.3202
    9 : "epochs=70_lr=0.01_temp=2_optim=adam_lr_sched=40_alpha=1_early_stop=True",

    #0.3204
    10 : "epochs=90_lr=0.01_temp=2_optim=adam_lr_sched=40_alpha=1.5_early_stop=True"
}