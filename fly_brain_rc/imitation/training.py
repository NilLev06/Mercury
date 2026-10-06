"""Recurrent supervised fitting and DAgger action mixing, not reward tuning."""
import torch


def teacher_probability(iteration):
    """Assistance follows checkpoint age, including recurring full routes."""
    if iteration < 6:
        return 1.0
    if iteration < 12:
        return .5
    if iteration < 24:
        return .2
    # Each pair covers both straight and cylinder training maps.
    return (1.0,1.0,.5,.5,0.0,0.0)[(iteration-24)%6]


def mix_action(teacher,student,beta):
    if not 0 <= beta <= 1:
        raise ValueError('invalid teacher mixing probability')
    action = teacher if torch.rand(()).item()<beta else student
    return action,teacher


def fit_demonstrations(model,optimizer,trajectories,*,updates=100,batch_size=8,
                       sequence_length=48,cancel=None):
    if not trajectories or any(len(x)!=len(y) or not len(x) for x,y in trajectories):
        raise ValueError('nonempty aligned demonstration sequences required')
    losses = []
    model.train()
    for _ in range(updates):
        if cancel and cancel():
            break
        inputs,labels = [],[]
        length = sequence_length
        burn = 24
        # Equal-length crops; context burn-in reconstructs recurrent state
        # without backpropagating through an entire several-minute episode.
        state_list = []
        for _ in range(batch_size):
            x,y = trajectories[int(torch.randint(len(trajectories),()))]
            crop_length = min(length,len(x))
            start = int(torch.randint(max(1,len(x)-crop_length+1),()))
            with torch.no_grad():
                state = (model(x[max(0,start-burn):start][None])[1] if start>0
                         else x.new_zeros(1,model.mask.shape[0]))
            state_list.append(state)
            inputs.append(x[start:start+crop_length])
            labels.append(y[start:start+crop_length])
        lengths = torch.tensor([len(values) for values in inputs])
        x = torch.nn.utils.rnn.pad_sequence(inputs,batch_first=True)
        y = torch.nn.utils.rnn.pad_sequence(labels,batch_first=True)
        optimizer.zero_grad()
        predicted,_ = model(x,torch.cat(state_list))
        # Stopping is rare in long drives: balance those demonstrations,
        # rather than inventing additional RL rewards.
        stop_weight = torch.where(y[...,0].abs()<.02,5.0,1.0)
        error = (predicted-y).square()*predicted.new_tensor([1.0,2.0])
        valid = torch.arange(x.shape[1])[None,:]<lengths[:,None]
        loss = (error.mean(-1)*stop_weight*valid).sum()/valid.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        losses.append(loss.item())
    return sum(losses)/len(losses) if losses else 0.0
