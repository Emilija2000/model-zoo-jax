from model_zoo_jax import load_multiple_datasets, load_nets, shuffle_data, model_save, model_restore
import chex
import haiku as hk
import jax
import jax.numpy as jnp

def test_zoo_loading(data_dir):
    '''use some dropped class dataset tot test'''
    for n in [1, 4, 5, 16, 18]:
        # exactly n networks read
        data,labels = load_nets(n, data_dir, flatten=False)
        assert len(data) == n
        assert len(labels['class_dropped']) == n
        assert len(labels['train/loss']) == n
        # read correctly + is tree
        chex.assert_tree_no_nones(data[0])
        # all read and flattened correctly
        data,labels = load_nets(n, data_dir, flatten=True)
        chex.assert_axis_dimension(data,0,n)
        chex.assert_equal_size([data[i] for i in data.shape[0]])
    
    for c in [1, 2, 4]:
        # exact number of models from the same subfolder
        data,labels = load_nets(16, data_dir, flatten=False, num_checkpoints=c)
        labels = labels['class_dropped']
        current = labels[0]
        c0 = c
        for n in range(16):
            if c0 > 0:
                assert current == labels[n]
                c0 = c0-1
            else:
                c0 = c
            current = labels[n]

def test_multizoo_loading(path_with_img_dataset1, path_with_img_dataset2):
    # all datasets are read in correct order
    n=5
    data, labels = load_multiple_datasets([path_with_img_dataset1,path_with_img_dataset2],n)
    labels = labels['dataset']
    
    for i in range(n):
        assert labels[i] == labels[0]
    chex.assert_trees_all_equal_sizes(data[:n])
    for i in range(n,2*n):
        assert labels[i] == labels[n]
    chex.assert_trees_all_equal_sizes(data[n:])

def test_shuffle(data_dir):
    # same rng same shuffle
    data,labels = load_nets(10, data_dir, flatten=False,num_checkpoints=1)
    data1,labels1 = shuffle_data(jax.random.PRNGKey(42),data,labels,chunks=1)
    data2,labels2 = shuffle_data(jax.random.PRNGKey(42),data,labels,chunks=1)
    chex.assert_equal(labels1,labels2)
    
    # chunked shuffle works correctly
    data,labels = load_nets(10, data_dir, flatten=False,num_checkpoints=4)
    data3,labels3 = shuffle_data(jax.random.PRNGKey(1),data,labels,chunks=4)
    labels = labels3['class_dropped']
    current = labels[0]
    c0 = 4
    for n in range(16):
        if c0 > 0:
            assert current == labels[n]
            c0 = c0-1
        else:
            c0 = 4
        current = labels[n]

def test_model_save_load():
    # the model is the same before saving and after restoring
    def net_fn(x):
        model = hk.Sequential([
            hk.Conv2D(output_channels=32, kernel_shape=3, stride=1, padding='SAME'),
            jax.nn.relu,
            hk.MaxPool(window_shape=(2, 2, 1), strides=(2, 2, 1), padding='SAME'),
            hk.Conv2D(output_channels=64, kernel_shape=3, stride=1, padding='SAME'),
            jax.nn.relu,
            hk.MaxPool(window_shape=(2, 2, 1), strides=(2, 2, 1), padding='SAME'),
            hk.Flatten(),
            hk.Linear(10),
        ])
        return model(x)

    net = hk.without_apply_rng(hk.transform(net_fn))
    model_pre_save = net.init(jax.random.PRNGKey(42), jnp.ones([1, 28, 28, 1]))
    model_save('savedir',model_pre_save)
    model_post_restore = model_restore('savedir')
    trees = [model_pre_save, model_post_restore]
    
    chex.assert_trees_all_equal_dtypes(model_pre_save, model_post_restore)
    chex.assert_tree_all_finite(model_post_restore)
    chex.assert_trees_all_equal_shapes(model_pre_save, model_post_restore)
    chex.assert_trees_all_close(*trees, rtol=1e-06)

if __name__=='__main__':
    test_zoo_loading()
    test_multizoo_loading()
    test_shuffle()
    test_model_save_load()
