class args():

    # training args
    epochs = 20
    batch_size  = 16
    train_num   = 40000

    # ── Dataset paths (KAIST flat structure) ──────────────────
    ir_train_dir = "/home/iec/vstung/TGFuse/dataset_flat/train/ir"
    vi_train_dir = "/home/iec/vstung/TGFuse/dataset_flat/train/vi"
    ir_val_dir   = "/home/iec/vstung/TGFuse/dataset_flat/val/ir"
    vi_val_dir   = "/home/iec/vstung/TGFuse/dataset_flat/val/vi"
    ir_test_dir  = "/home/iec/vstung/TGFuse/dataset_flat/test/ir"
    vi_test_dir  = "/home/iec/vstung/TGFuse/dataset_flat/test/vi"

    # legacy alias (dùng trong train.py gốc)
    dataset2     = ir_train_dir

    HEIGHT = 256
    WIDTH  = 256

    save_model_dir = "/home/iec/vstung/TGFuse/models"
    save_loss_dir  = "/home/iec/vstung/TGFuse/models/loss"

    image_size = 256
    cuda       = 1
    seed       = 42

    ssim_path  = ['1e0', '1e1', '1e2', '1e3', '1e4']
    alpha = 0.5
    beta  = 0.5
    gama  = 1
    yita  = 1
    deta  = 1

    lr          = 1e-4
    lr_d        = 1e-4
    log_interval = 10

    resume          = None
    trans_model_path = None
    is_para          = False