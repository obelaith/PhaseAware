import torch
import torch.nn as nn

from phaseaware.optimizer import PhaseAwareOptimizer

def set_seed(seed=42):
    torch.manual_seed(seed)


# ---------------------------------------------------------
# Test 1: Toy quadratic convergence
# ---------------------------------------------------------

def test_quadratic_convergence():

    set_seed()

    x = torch.nn.Parameter(
        torch.tensor([10.0])
    )

    optimizer = PhaseAwareOptimizer(
        [x],
        lr_max=0.1,
        lr_min=1e-3,
        total_steps=500,
        noise_max=0.01
    )


    initial_distance = abs(x.item())


    for _ in range(500):

        optimizer.zero_grad()

        loss = x.pow(2)

        loss.backward()

        optimizer.step()



    final_distance = abs(x.item())


    print(
        "Quadratic optimization:"
    )

    print(
        "Initial:",
        initial_distance
    )

    print(
        "Final:",
        final_distance
    )


    assert final_distance < initial_distance

    assert final_distance < 0.5



# ---------------------------------------------------------
# Test 2: State dict save/load
# ---------------------------------------------------------

def test_state_dict():

    set_seed()


    model1 = nn.Linear(5, 2)

    opt1 = PhaseAwareOptimizer(
        model1.parameters()
    )


    x = torch.randn(4,5)

    y = torch.randn(4,2)


    opt1.zero_grad()

    loss = (
        model1(x)-y
    ).pow(2).mean()

    loss.backward()

    opt1.step()



    checkpoint = {
        "model": model1.state_dict(),
        "optimizer": opt1.state_dict()
    }



    model2 = nn.Linear(5,2)

    opt2 = PhaseAwareOptimizer(
        model2.parameters()
    )


    model2.load_state_dict(
        checkpoint["model"]
    )

    opt2.load_state_dict(
        checkpoint["optimizer"]
    )



    for p1, p2 in zip(
        model1.parameters(),
        model2.parameters()
    ):

        assert torch.equal(
            p1,
            p2
        )


    print(
        "State dict test passed"
    )



# ---------------------------------------------------------
# Test 3: Compare with Adam
# ---------------------------------------------------------

def train_optimizer(optimizer_name):

    set_seed()


    model = nn.Linear(10,1)


    if optimizer_name == "PhaseAware":

        optimizer = PhaseAwareOptimizer(
            model.parameters(),
            lr_max=1e-2,
            lr_min=1e-4,
            total_steps=300
        )

    else:

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-3
        )


    x = torch.randn(100,10)

    true_w = torch.randn(10,1)

    y = x @ true_w



    losses = []


    for _ in range(300):

        optimizer.zero_grad()


        pred = model(x)


        loss = (
            pred-y
        ).pow(2).mean()


        loss.backward()

        optimizer.step()


        losses.append(
            loss.item()
        )


    return losses[-1]



def test_compare_adam():

    phaseaware_loss = train_optimizer(
        "PhaseAware"
    )

    adam_loss = train_optimizer(
        "Adam"
    )


    print(
        "PhaseAware final loss:",
        phaseaware_loss
    )

    print(
        "Adam final loss:",
        adam_loss
    )


    assert phaseaware_loss < 1.0
    assert adam_loss < 10.0



# ---------------------------------------------------------
# Run tests manually
# ---------------------------------------------------------

if __name__ == "__main__":

    test_quadratic_convergence()

    test_state_dict()

    test_compare_adam()

    print(
        "\nAll tests passed."
    )