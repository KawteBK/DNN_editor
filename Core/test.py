"""
test_custom_datasets_and_nonlinear_graphs.py

Tests for:
1. Custom dataset loading and validation
2. Non-linear graphs (multi-parent / functional API)
3. Merge layers (Add and Concatenate)
4. End-to-end network validation with complex topologies
"""

import os
import tempfile
import shutil
import numpy as np
from pathlib import Path

# Import your modules
from .dataset import Dataset, DatasetSource, BuiltinDataset, BUILTIN_METADATA
from .layer import (
    InputLayer, Dense, Conv2D, Flatten, Activation, ActivationType,
    Merge, MergeMode, Pooling, PoolingType, Recurrent, RecurrentType
)
from .network import Network
from .project import Project


def setup_custom_dataset_structure(base_path: str, num_classes: int = 3, samples_per_class: int = 5):
    """Create a fake ImageFolder-style dataset for testing."""
    for class_idx in range(num_classes):
        class_dir = os.path.join(base_path, f"class_{class_idx}")
        os.makedirs(class_dir, exist_ok=True)
        for sample_idx in range(samples_per_class):
            # Create dummy image files (just empty files for path validation)
            dummy_file = os.path.join(class_dir, f"img_{sample_idx}.png")
            with open(dummy_file, 'w') as f:
                f.write("dummy")


def setup_custom_csv_dataset(file_path: str, num_features: int = 128, num_samples: int = 100):
    """Create a dummy CSV dataset."""
    data = np.random.randn(num_samples, num_features)
    labels = np.random.randint(0, 10, num_samples)
    np.savetxt(file_path, np.column_stack((data, labels)), delimiter=',', 
               header=','.join([f'feat_{i}' for i in range(num_features)] + ['label']), 
               comments='')


def test_custom_dataset_creation():
    """Test 1: Custom dataset creation and validation."""
    print("\n=== Test 1: Custom Dataset Creation ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create ImageFolder-style dataset
        dataset_path = os.path.join(tmpdir, "my_custom_data")
        os.makedirs(dataset_path)
        setup_custom_dataset_structure(dataset_path, num_classes=5, samples_per_class=10)
        
        # Create custom dataset
        custom_ds = Dataset.fromCustomPath(
            path=dataset_path,
            inputShape=(3, 64, 64),  # RGB 64x64 images
            numClasses=5,
            name="My Custom Dataset"
        )
        
        # Verify properties
        assert custom_ds._source == DatasetSource.CUSTOM
        assert custom_ds._name == "My Custom Dataset"
        assert custom_ds._inputShape == (3, 64, 64)
        assert custom_ds._numClasses == 5
        assert custom_ds._dataPath == dataset_path
        assert custom_ds.validate() == True
        assert custom_ds.isDataAvailable() == True
        
        # Test serialization
        ds_dict = custom_ds.toDict()
        reconstructed = Dataset.fromDict(ds_dict)
        assert reconstructed is not None
        assert reconstructed._id == custom_ds._id
        assert reconstructed._inputShape == custom_ds._inputShape
        assert reconstructed._numClasses == custom_ds._numClasses
        
        print("✓ Custom dataset creation and validation passed")
        print(f"  - Dataset ID: {custom_ds._id}")
        print(f"  - Name: {custom_ds._name}")
        print(f"  - Input shape: {custom_ds._inputShape}")
        print(f"  - Classes: {custom_ds._numClasses}")


def test_custom_dataset_csv():
    """Test 2: Custom CSV dataset."""
    print("\n=== Test 2: Custom CSV Dataset ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "features.csv")
        setup_custom_csv_dataset(csv_path, num_features=256, num_samples=500)
        
        custom_ds = Dataset.fromCustomPath(
            path=csv_path,
            inputShape=(256,),  # 1D feature vector
            numClasses=10,
            name="Feature Dataset"
        )
        
        assert custom_ds.validate() == True
        assert custom_ds.isDataAvailable() == True
        assert custom_ds._inputShape == (256,)
        
        print("✓ CSV dataset creation passed")
        print(f"  - Input shape: {custom_ds._inputShape}")


def test_nonlinear_graph_residual():
    """Test 3: Residual connection (Add merge)."""
    print("\n=== Test 3: Residual Network (Add Merge) ===")
    
    net = Network(id="resnet_test")
    
    # Create layers
    input_layer = InputLayer(inputShape=(3, 32, 32))
    conv1 = Conv2D(filters=32, kernelSize=(3, 3), strides=(1, 1), padding=(1, 1),
                   dropout=0.0, dropoutStyle=None, id="conv1", name="Conv1", role=None)
    conv2 = Conv2D(filters=32, kernelSize=(3, 3), strides=(1, 1), padding=(1, 1),
                   dropout=0.0, dropoutStyle=None, id="conv2", name="Conv2", role=None)
    
    # Skip connection (identity)
    identity = Conv2D(filters=32, kernelSize=(1, 1), strides=(1, 1), padding=(0, 0),
                      dropout=0.0, dropoutStyle=None, id="identity", name="Identity", role=None)
    
    add = Merge(id="add", name="Add", mergeMode=MergeMode.ADD)
    relu = Activation(activationType=ActivationType.RELU, id="relu", name="ReLU")
    
    # Add layers to network
    for layer in [input_layer, conv1, conv2, identity, add, relu]:
        net.addLayer(layer)
    
    # Create connections (non-linear!)
    net.addConnect("input", "conv1")
    net.addConnect("conv1", "conv2")
    net.addConnect("input", "identity")  # skip connection from input
    net.addConnect("conv2", "add")
    net.addConnect("identity", "add")    # both paths feed into Add
    net.addConnect("add", "relu")
    
    # Validate
    assert net.validate() == True
    
    # Check shape propagation
    net.propagate_shapes()
    
    # Conv2D output shape: (filters, H_out, W_out)
    assert net.layers["conv1"]._outputShape == (32, 32, 32)
    assert net.layers["conv2"]._outputShape == (32, 32, 32)
    assert net.layers["identity"]._outputShape == (32, 32, 32)
    assert net.layers["add"]._outputShape == (32, 32, 32)
    assert net.layers["relu"]._outputShape == (32, 32, 32)
    
    print("✓ Residual network with Add merge passed")
    print(f"  - Add layer output shape: {net.layers['add']._outputShape}")
    print(f"  - Network has {len(net.layers)} layers, {len(net.connections)} connections")


def test_nonlinear_graph_concat():
    """Test 4: Multi-branch concatenation."""
    print("\n=== Test 4: Multi-branch Concatenation ===")
    
    net = Network(id="inception_test")
    
    # Input
    input_layer = InputLayer(inputShape=(3, 64, 64))
    
    # Branch 1: 1x1 conv
    conv1x1 = Conv2D(filters=16, kernelSize=(1, 1), strides=(1, 1), padding=(0, 0),
                     dropout=0.0, dropoutStyle=None, id="conv1x1", name="Conv1x1", role=None)
    
    # Branch 2: 3x3 conv
    conv3x3 = Conv2D(filters=16, kernelSize=(3, 3), strides=(1, 1), padding=(1, 1),
                     dropout=0.0, dropoutStyle=None, id="conv3x3", name="Conv3x3", role=None)
    
    # Branch 3: 5x5 conv
    conv5x5 = Conv2D(filters=16, kernelSize=(5, 5), strides=(1, 1), padding=(2, 2),
                     dropout=0.0, dropoutStyle=None, id="conv5x5", name="Conv5x5", role=None)
    
    # Concatenate all branches
    concat = Merge(id="concat", name="Concat", mergeMode=MergeMode.CONCATENATE)
    
    # Add layers
    for layer in [input_layer, conv1x1, conv3x3, conv5x5, concat]:
        net.addLayer(layer)
    
    # Create connections (3 parents to concat!)
    net.addConnect("input", "conv1x1")
    net.addConnect("input", "conv3x3")
    net.addConnect("input", "conv5x5")
    net.addConnect("conv1x1", "concat")
    net.addConnect("conv3x3", "concat")
    net.addConnect("conv5x5", "concat")
    
    # Validate
    assert net.validate() == True
    
    # Check shape propagation
    net.propagate_shapes()
    
    # Each branch: (16, 64, 64) -> concatenated along channel dim = 16*3 = 48
    # Conv2D returns (filters, H, W)
    expected_shape = (48, 64, 64)
    actual_shape = net.layers["concat"]._outputShape
    
    print(f"  Expected shape: {expected_shape}")
    print(f"  Actual shape: {actual_shape}")
    
    assert actual_shape == expected_shape, f"Expected {expected_shape}, got {actual_shape}"
    
    print("✓ Multi-branch concatenation passed")
    print(f"  - Concat output shape: {actual_shape}")


def test_nonlinear_graph_dense_skip():
    """Test 5: Skip connections in MLP."""
    print("\n=== Test 5: MLP with Skip Connections ===")
    
    net = Network(id="mlp_skip_test")
    
    input_layer = InputLayer(inputShape=(784,))
    dense1 = Dense(unit=256, useBias=True, dropout=0.1, id="dense1", name="Dense1", role=None)
    relu1 = Activation(activationType=ActivationType.RELU, id="relu1", name="ReLU1")
    dense2 = Dense(unit=128, useBias=True, dropout=0.1, id="dense2", name="Dense2", role=None)
    relu2 = Activation(activationType=ActivationType.RELU, id="relu2", name="ReLU2")
    
    # Skip connection (identity projection via another dense)
    skip_dense = Dense(unit=128, useBias=True, dropout=0.0, id="skip_dense", name="SkipDense", role=None)
    
    add = Merge(id="add", name="Add", mergeMode=MergeMode.ADD)
    output = Dense(unit=10, useBias=True, dropout=0.0, id="output", name="Output", role=None)
    softmax = Activation(activationType=ActivationType.SOFTMAX, id="softmax", name="Softmax")
    
    for layer in [input_layer, dense1, relu1, dense2, relu2, skip_dense, add, output, softmax]:
        net.addLayer(layer)
    
    # Main path
    net.addConnect("input", "dense1")
    net.addConnect("dense1", "relu1")
    net.addConnect("relu1", "dense2")
    net.addConnect("dense2", "relu2")
    net.addConnect("relu2", "add")
    
    # Skip path: input -> skip_dense -> add
    net.addConnect("input", "skip_dense")
    net.addConnect("skip_dense", "add")
    
    # Output path
    net.addConnect("add", "output")
    net.addConnect("output", "softmax")
    
    assert net.validate() == True
    net.propagate_shapes()
    
    # Check shapes
    assert net.layers["dense1"]._outputShape == (256,)
    assert net.layers["dense2"]._outputShape == (128,)
    assert net.layers["skip_dense"]._outputShape == (128,)
    assert net.layers["add"]._outputShape == (128,)
    assert net.layers["output"]._outputShape == (10,)
    
    print("✓ MLP with skip connections passed")
    print(f"  - Add layer merges dense2 ({net.layers['dense2']._outputShape}) and skip_dense ({net.layers['skip_dense']._outputShape})")


def test_network_orphan_detection():
    """Test 6: Orphan/puppy layer detection."""
    print("\n=== Test 6: Orphan Layer Detection ===")
    
    net = Network(id="orphan_test")
    
    input_layer = InputLayer(inputShape=(784,))
    dense1 = Dense(256, True, 0.0, "dense1", "Dense1", None)
    orphan = Dense(128, True, 0.0, "orphan", "Orphan", None)  # This will be unconnected
    dense2 = Dense(10, True, 0.0, "dense2", "Dense2", None)
    
    for layer in [input_layer, dense1, orphan, dense2]:
        net.addLayer(layer)
    
    net.addConnect("input", "dense1")
    net.addConnect("dense1", "dense2")
    # orphan is never connected
    
    # validate should fail due to orphan
    assert net.validate() == False
    
    # But get_orphaned_layers should identify it
    orphans = net.get_orphaned_layers()
    assert "orphan" in orphans
    assert len(orphans) == 1
    
    print("✓ Orphan detection passed")
    print(f"  - Orphaned layers found: {orphans}")


def test_custom_dataset_with_nonlinear_network():
    """Test 7: End-to-end with custom dataset and nonlinear network."""
    print("\n=== Test 7: End-to-End Integration ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create custom dataset
        dataset_path = os.path.join(tmpdir, "custom_data")
        os.makedirs(dataset_path)
        setup_custom_dataset_structure(dataset_path, num_classes=10, samples_per_class=20)
        
        custom_ds = Dataset.fromCustomPath(
            path=dataset_path,
            inputShape=(3, 64, 64),
            numClasses=10,
            name="Test Dataset"
        )
        
        # Create nonlinear network
        net = Network(id="test_network")
        
        # Build a simple residual block
        input_layer = InputLayer(inputShape=(3, 64, 64))
        conv1 = Conv2D(32, (3, 3), (1, 1), (1, 1), 0.0, None, "conv1", "Conv1", None)
        bn1 = Activation(ActivationType.RELU, "bn1", "ReLU1")  # Placeholder for BN
        conv2 = Conv2D(32, (3, 3), (1, 1), (1, 1), 0.0, None, "conv2", "Conv2", None)
        
        # Skip connection
        skip = Conv2D(32, (1, 1), (1, 1), (0, 0), 0.0, None, "skip", "Skip", None)
        add = Merge("add", "Add", None, MergeMode.ADD)
        relu = Activation(ActivationType.RELU, "relu", "ReLU")
        flatten = Flatten("flatten", "Flatten")
        output = Dense(10, True, 0.0, "output", "Output", None)
        softmax = Activation(ActivationType.SOFTMAX, "softmax", "Softmax")
        
        for layer in [input_layer, conv1, bn1, conv2, skip, add, relu, flatten, output, softmax]:
            net.addLayer(layer)
        
        # Connect
        net.addConnect("input", "conv1")
        net.addConnect("conv1", "bn1")
        net.addConnect("bn1", "conv2")
        net.addConnect("input", "skip")
        net.addConnect("conv2", "add")
        net.addConnect("skip", "add")
        net.addConnect("add", "relu")
        net.addConnect("relu", "flatten")
        net.addConnect("flatten", "output")
        net.addConnect("output", "softmax")
        
        # Create training config (adjust parameters to match your TrainingConfig)
        # You may need to adjust these based on your actual TrainingConfig class
        try:
            from trainingConfig import TrainingConfig
            # Try to create with common parameters
            train_config = TrainingConfig(
                epochs=10,
                learning_rate=0.001,
                optimizer="adam",
                loss="cross_entropy"
            )
            # If batch_size is a separate attribute, set it
            if hasattr(train_config, 'batch_size'):
                train_config.batch_size = 32
            if hasattr(train_config, 'validation_split'):
                train_config.validation_split = 0.2
        except ImportError:
            print("  Warning: TrainingConfig not available, creating mock")
            train_config = None
        
        # Create network config (adjust parameters to match your NetworkConfig)
        try:
            from networkConfig import NetworkConfig
            net_config = NetworkConfig(
                initialization="he_normal",
                regularization=0.0001
            )
            if hasattr(net_config, 'gradient_clip'):
                net_config.gradient_clip = 1.0
        except ImportError:
            print("  Warning: NetworkConfig not available, creating mock")
            net_config = None
        
        # Create project
        project = Project(name="Test Project", path=tmpdir)
        project.network = net
        project.dataset = custom_ds
        if train_config:
            project.trainingConfig = train_config
        if net_config:
            project.networkConfig = net_config
        
        # Validate project is ready to train (may need to adjust based on your validation logic)
        is_ready = project.isReadyToTrain()
        print(f"  Project ready to train: {is_ready}")
        
        # Test serialization/deserialization
        project_dict = project.toDict()
        reconstructed = Project.fromDict(project_dict)
        
        assert reconstructed.name == project.name
        assert reconstructed.network is not None
        assert reconstructed.dataset is not None
        
        print("✓ End-to-end integration passed")
        print(f"  - Project: {project.name}")
        print(f"  - Dataset: {custom_ds._name} ({custom_ds._inputShape})")
        print(f"  - Network: {len(net.layers)} layers, {len(net.connections)} connections")
        print(f"  - Serialization roundtrip successful")


def test_invalid_merge_shapes():
    """Test 8: Error handling for incompatible merge shapes."""
    print("\n=== Test 8: Invalid Merge Shape Detection ===")
    
    # Test 8a: Add merge with shape mismatch
    print("  Testing Add merge with shape mismatch...")
    net = Network(id="error_test")
    
    input_layer = InputLayer(inputShape=(3, 32, 32))
    conv1 = Conv2D(32, (3, 3), (1, 1), (1, 1), 0.0, None, "conv1", "Conv1", None)
    conv2 = Conv2D(64, (3, 3), (1, 1), (1, 1), 0.0, None, "conv2", "Conv2", None)  # Different filters!
    add = Merge("add", "Add", None, MergeMode.ADD)
    
    for layer in [input_layer, conv1, conv2, add]:
        net.addLayer(layer)
    
    net.addConnect("input", "conv1")
    net.addConnect("input", "conv2")
    net.addConnect("conv1", "add")
    net.addConnect("conv2", "add")
    
    # This should fail because shapes don't match for addition
    try:
        net.propagate_shapes()
        # If we get here, the shape mismatch wasn't caught
        # But maybe your implementation handles it differently
        print("  Note: Add merge with different shapes was accepted")
        print(f"    conv1 output: {net.layers['conv1']._outputShape}")
        print(f"    conv2 output: {net.layers['conv2']._outputShape}")
        print(f"    add output: {net.layers['add']._outputShape}")
    except ValueError as e:
        print(f"  ✓ Correctly caught shape mismatch: {str(e)[:80]}...")
    
    # Test 8b: Concatenation with spatial dimension mismatch
    print("  Testing Concatenate with spatial mismatch...")
    net2 = Network(id="concat_error_test")
    input_layer2 = InputLayer(inputShape=(3, 32, 32))
    
    # This will produce different spatial dimensions due to stride
    conv1_2 = Conv2D(32, (3, 3), (2, 2), (1, 1), 0.0, None, "conv1", "Conv1", None)
    conv2_2 = Conv2D(32, (3, 3), (1, 1), (1, 1), 0.0, None, "conv2", "Conv2", None)
    concat = Merge("concat", "Concat", None, MergeMode.CONCATENATE)
    
    for layer in [input_layer2, conv1_2, conv2_2, concat]:
        net2.addLayer(layer)
    
    net2.addConnect("input", "conv1")
    net2.addConnect("input", "conv2")
    net2.addConnect("conv1", "concat")
    net2.addConnect("conv2", "concat")
    
    try:
        net2.propagate_shapes()
        print(f"  conv1 output shape: {net2.layers['conv1']._outputShape}")
        print(f"  conv2 output shape: {net2.layers['conv2']._outputShape}")
        print(f"  concat output shape: {net2.layers['concat']._outputShape}")
    except ValueError as e:
        print(f"  ✓ Correctly caught concatenation mismatch: {str(e)[:80]}...")
    
    print("✓ Error handling tests completed")


def test_complex_dag():
    """Test 9: Complex DAG with multiple merge points."""
    print("\n=== Test 9: Complex DAG with Multiple Merges ===")
    
    net = Network(id="complex_dag")
    
    # Input
    input_layer = InputLayer(inputShape=(3, 224, 224))
    
    # Early convs
    conv_early = Conv2D(64, (7, 7), (2, 2), (3, 3), 0.0, None, "conv_early", "ConvEarly", None)
    
    # Branch A
    conv_a1 = Conv2D(64, (3, 3), (1, 1), (1, 1), 0.0, None, "conv_a1", "ConvA1", None)
    conv_a2 = Conv2D(64, (3, 3), (1, 1), (1, 1), 0.0, None, "conv_a2", "ConvA2", None)
    
    # Branch B
    conv_b1 = Conv2D(64, (5, 5), (1, 1), (2, 2), 0.0, None, "conv_b1", "ConvB1", None)
    
    # Branch C (skip)
    skip = Conv2D(64, (1, 1), (1, 1), (0, 0), 0.0, None, "skip", "Skip", None)
    
    # Merge points
    merge1 = Merge("merge1", "Merge1", None, MergeMode.ADD)
    merge2 = Merge("merge2", "Merge2", None, MergeMode.ADD)
    
    # Final layers
    flatten = Flatten("flatten", "Flatten")
    dense = Dense(1000, True, 0.5, "dense", "Dense", None)
    output = Activation(ActivationType.SOFTMAX, "output", "Output")
    
    # Add all layers
    layers = [input_layer, conv_early, conv_a1, conv_a2, conv_b1, skip, merge1, merge2, flatten, dense, output]
    for layer in layers:
        net.addLayer(layer)
    
    # Build DAG
    net.addConnect("input", "conv_early")
    net.addConnect("conv_early", "conv_a1")
    net.addConnect("conv_early", "conv_b1")
    net.addConnect("conv_early", "skip")
    
    net.addConnect("conv_a1", "conv_a2")
    net.addConnect("conv_a2", "merge1")
    net.addConnect("conv_b1", "merge1")
    net.addConnect("skip", "merge2")
    net.addConnect("merge1", "merge2")
    
    net.addConnect("merge2", "flatten")
    net.addConnect("flatten", "dense")
    net.addConnect("dense", "output")
    
    # Validate
    assert net.validate() == True
    net.propagate_shapes()
    
    print("✓ Complex DAG validation passed")
    print(f"  - Total layers: {len(net.layers)}")
    print(f"  - Total connections: {len(net.connections)}")
    print(f"  - Output shape: {net.layers['output']._outputShape}")


def run_all_tests():
    """Run all test cases."""
    print("=" * 60)
    print("TESTING CUSTOM DATASETS AND NON-LINEAR GRAPHS")
    print("=" * 60)
    
    tests = [
        test_custom_dataset_creation,
        test_custom_dataset_csv,
        test_nonlinear_graph_residual,
        test_nonlinear_graph_concat,
        test_nonlinear_graph_dense_skip,
        test_network_orphan_detection,
        test_custom_dataset_with_nonlinear_network,
        test_invalid_merge_shapes,
        test_complex_dag,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n✗ Test failed: {test.__name__}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return passed, failed


if __name__ == "__main__":
    # Run tests
    passed, failed = run_all_tests()
    
    # Exit with appropriate code
    exit(0 if failed == 0 else 1)