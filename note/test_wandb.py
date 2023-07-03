import wandb


def run():
    wandb.login()

    wandb.init(
        project='Demo',
        name='first run',
        config={
            'machine': 'Dell'
        }
    )

    batch_size = 30
    total_epoch = 10

    loss = 300
    iter = 1
    for i in range(total_epoch):
        for j in range(batch_size):
            wandb.log({
                'iter_loss': loss / iter
            })
            iter += 1
        wandb.log({
            'bath_loss': loss / iter
        })


if __name__ == '__main__':
    run()
