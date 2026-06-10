import os
import math
import numpy as np
from tqdm import tqdm
from scipy import linalg
import torch


# (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)  # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)  # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)  # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists


def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
        #         print(correct_vec, bool_mat[:, i])
        correct_vec = (correct_vec | bool_mat[:, i])
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    matching_score = dist_mat.trace()
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0), matching_score
    else:
        return top_k_mat, matching_score


def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()


def calculate_diversity(activation, diversity_times, emb_scale=1, div_scale=1):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    activation = activation * emb_scale
    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm((activation[first_indices] - activation[second_indices]) / div_scale, axis=1)
    return dist.mean()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(activations, emb_scale=1):
    activations = activations * emb_scale
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_frechet_feature_distance(feature_list1, feature_list2):
    feature_list1 = np.stack(feature_list1)
    feature_list2 = np.stack(feature_list2)

    # normalize the scale
    mean = np.mean(feature_list1, axis=0)
    std = np.std(feature_list1, axis=0) + 1e-10
    feature_list1 = (feature_list1 - mean) / std
    feature_list2 = (feature_list2 - mean) / std

    dist = calculate_frechet_distance(
        mu1=np.mean(feature_list1, axis=0),
        sigma1=np.cov(feature_list1, rowvar=False),
        mu2=np.mean(feature_list2, axis=0),
        sigma2=np.cov(feature_list2, rowvar=False),
    )
    return dist


def predict_interx(joint_dist, motions, motion_lens, net, dataset):
    bs, t, d = motions.shape
    pred_pose_eval = np.zeros_like(motions)
    for i in range(bs):
        s_seq = motion_lens[i]
        single_motion = motions[i:i + 1, :s_seq].cpu().numpy()
        single_motion = single_motion.reshape(1, s_seq, 56, -1)
        m_dim = single_motion.shape[-1]
        assert m_dim == 12
        single_motion1, single_motion2 = single_motion[..., :m_dim // 2], single_motion[..., m_dim // 2:]
        single_motion1 = single_motion1.reshape(1, s_seq, -1)
        single_motion2 = single_motion2.reshape(1, s_seq, -1)
        
        single_motion1 = dataset.normalizer.forward(single_motion1)
        single_motion2 = dataset.normalizer.forward(single_motion2)
        single_motion = np.concatenate([single_motion1, single_motion2], axis=-1)
        single_motion = torch.from_numpy(single_motion).cuda().float()

        single_joint_dist = joint_dist[i:i + 1, :s_seq].cpu().numpy()
        single_joint_dist = dataset.normalizer.forward_dist(single_joint_dist)
        single_joint_dist = torch.from_numpy(single_joint_dist).cuda().float()

        # predict
        pred_pose1, pred_pose2 = net(single_motion, single_joint_dist)[:2]
        pred_pose1 = pred_pose1.cpu().numpy()
        pred_pose1 = dataset.normalizer.backward(pred_pose1)
        pred_pose1 = pred_pose1.reshape(1, s_seq, 56, -1)

        pred_pose2 = pred_pose2.cpu().numpy()
        pred_pose2 = dataset.normalizer.backward(pred_pose2)
        pred_pose2 = pred_pose2.reshape(1, s_seq, 56, -1)

        pred_pose = np.concatenate([pred_pose1, pred_pose2], axis=-1)
        pred_pose = pred_pose.reshape(1, s_seq, -1)
        pred_pose_eval[i:i + 1, :motion_lens[i], :d] = pred_pose

    pred_pose_eval = torch.from_numpy(pred_pose_eval).cuda()
    return pred_pose_eval


def predict_inter_human(motion_norm, joint_dist, net, motion_lens, dataset):
    # normalize motion for model prediction
    bs, seq, dim = motion_norm.shape
    motion_norm = motion_norm.detach().float().cuda()
    joint_dist = joint_dist.detach().float().cuda()
    assert dim == 524

    pred_pose_eval = np.zeros((bs, seq, dim))
    for i in range(bs):
        pred_pose1, pred_pose2 = net(
            motion_norm[i:i + 1, :motion_lens[i]],
            joint_dist[i:i + 1, :motion_lens[i]],
        )[:2]
        pred_pose1 = pred_pose1.cpu().numpy()
        pred_pose2 = pred_pose2.cpu().numpy()

        pred_pose1 = dataset.normalizer.backward(pred_pose1)
        pred_pose2 = dataset.normalizer.backward(pred_pose2)
        pred_pose = np.concatenate([pred_pose1, pred_pose2], axis=-1)
        pred_pose_eval[i:i + 1, :motion_lens[i], :dim] = pred_pose

    pred_mot1, pred_mot2 = pred_pose_eval[..., :dim // 2], pred_pose_eval[..., dim // 2:]
    pred_mot1, pred_mot2 = torch.from_numpy(pred_mot1), torch.from_numpy(pred_mot2)
    return pred_mot1, pred_mot2


@torch.no_grad()
def evaluation_vqvae(
    dataname,
    out_dir,
    val_loader,
    net,
    logger,
    writer,
    nb_iter,
    best_fid,
    best_iter,
    best_div,
    best_top1,
    best_top2,
    best_top3,
    best_matching,
    eval_wrapper,
    draw=True,
    save=True,
    diversity_times=300,
    mm_num_repeats=30,
    mm_num_times=10,
):
    # ! Scaling is required for the evaluation of the InterHuman dataset.
    EMB_SCALE = 6 if dataname == 'InterHuman' else 1
    DIV_SCALE = 2 if dataname == 'InterHuman' else 1

    net.eval()
    dataset = val_loader.dataset

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    for batch in val_loader:
        if dataname == 'InterHuman':
            name, text, motion1, motion2, motion_norm, joint_dist, motion_lens = batch

            # unnormalize motion for eval wrapper
            bs = motion_norm.shape[0]
            et, em = eval_wrapper.get_co_embeddings([name, text, motion1, motion2, motion_lens])
            
            pred_mot1, pred_mot2 = predict_inter_human(motion_norm, joint_dist, net, motion_lens, dataset)
            et_pred, em_pred = eval_wrapper.get_co_embeddings([name, text, pred_mot1, pred_mot2, motion_lens])

        elif dataname == 'InterX':
            word_embeddings, pos_one_hots, caption, sent_lens, motions, motion_reset, joint_dist, motion_lens, token = batch
            # unnormalize motion for eval wrapper
            bs = motions.shape[0]
            et, em = eval_wrapper.get_co_embeddings(
                word_embs=word_embeddings,
                pos_ohot=pos_one_hots,
                cap_lens=sent_lens,
                motions=motions,
                m_lens=motion_lens,
            )

            pred_pose_eval = predict_interx(
                joint_dist,
                # motions,  # for UnReset
                motion_reset,
                motion_lens,
                net,
                dataset
            )

            et_pred, em_pred = eval_wrapper.get_co_embeddings(
                word_embs=word_embeddings,
                pos_ohot=pos_one_hots,
                cap_lens=sent_lens,
                motions=pred_pose_eval,
                m_lens=motion_lens,
            )
        else:
            raise ValueError('Unknown dataset')

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np, EMB_SCALE)
    mu, cov = calculate_activation_statistics(motion_pred_np, EMB_SCALE)

    diversity_real = calculate_diversity(motion_annotation_np, diversity_times if nb_sample > diversity_times else 100, EMB_SCALE, DIV_SCALE)
    diversity = calculate_diversity(motion_pred_np, diversity_times if nb_sample > diversity_times else 100, EMB_SCALE, DIV_SCALE)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    logger.info(msg)

    if draw:
        writer.add_scalar('Test/FID', fid, nb_iter)
        writer.add_scalar('Test/Diversity', diversity, nb_iter)
        writer.add_scalar('Test/top1', R_precision[0], nb_iter)
        writer.add_scalar('Test/top2', R_precision[1], nb_iter)
        writer.add_scalar('Test/top3', R_precision[2], nb_iter)
        writer.add_scalar('Test/matching_score', matching_score_pred, nb_iter)

    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        logger.info(msg)
        best_fid, best_iter = fid, nb_iter
        if save:
            torch.save({'net': net.state_dict(), 'nb_iter': nb_iter}, os.path.join(out_dir, 'net_best_fid.pth'))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        logger.info(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        logger.info(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        logger.info(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        logger.info(msg)
        best_top3 = R_precision[2]

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        logger.info(msg)
        best_matching = matching_score_pred

    if save:
        torch.save({'net': net.state_dict(), 'nb_iter': nb_iter}, os.path.join(out_dir, 'net_last.pth'))

    net.train()
    return best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger


@torch.no_grad()
def evaluation_vqvae_test(
    dataname,
    val_loader,
    net,
    repeat_id,
    eval_wrapper,
    diversity_times=300,
    mm_num_repeats=30,
    mm_num_times=10,
):
    # ! Scaling is required for the evaluation of the InterHuman dataset.
    EMB_SCALE = 6 if dataname == 'InterHuman' else 1
    DIV_SCALE = 2 if dataname == 'InterHuman' else 1

    net.eval()
    dataset = val_loader.dataset

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    for batch in val_loader:
        if dataname == 'InterHuman':
            name, text, motion1, motion2, motion_norm, joint_dist, motion_lens = batch

            # unnormalize motion for eval wrapper
            bs = motion_norm.shape[0]
            et, em = eval_wrapper.get_co_embeddings([name, text, motion1, motion2, motion_lens])
            
            pred_mot1, pred_mot2 = predict_inter_human(motion_norm, joint_dist, net, motion_lens, dataset)
            et_pred, em_pred = eval_wrapper.get_co_embeddings([name, text, pred_mot1, pred_mot2, motion_lens])

        elif dataname == 'InterX':
            word_embeddings, pos_one_hots, caption, sent_lens, motions, motion_reset, joint_dist, motion_lens, token = batch

            # unnormalize motion for eval wrapper
            bs, seq, dim = motions.shape
            et, em = eval_wrapper.get_co_embeddings(
                word_embs=word_embeddings,
                pos_ohot=pos_one_hots,
                cap_lens=sent_lens,
                motions=motions,
                m_lens=motion_lens,
            )
            
            pred_pose_eval = predict_interx(
                joint_dist,
                # motions,  # for UnReset
                motion_reset,
                motion_lens,
                net,
                dataset
            )

            et_pred, em_pred = eval_wrapper.get_co_embeddings(
                word_embs=word_embeddings,
                pos_ohot=pos_one_hots,
                cap_lens=sent_lens,
                motions=pred_pose_eval,
                m_lens=motion_lens,
            )
        else:
            raise ValueError('Unknown dataset')

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np, EMB_SCALE)
    mu, cov = calculate_activation_statistics(motion_pred_np, EMB_SCALE)

    diversity_real = calculate_diversity(motion_annotation_np, diversity_times if nb_sample > diversity_times else 100, EMB_SCALE, DIV_SCALE)
    diversity = calculate_diversity(motion_pred_np, diversity_times if nb_sample > diversity_times else 100, EMB_SCALE, DIV_SCALE)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Re {repeat_id} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    print(msg)
    return fid, diversity, R_precision, matching_score_pred


@torch.no_grad()
def evaluation_vqvae_test_real(dataname, val_loader, repeat_id, eval_wrapper, diversity_times=300, mm_num_repeats=30, mm_num_times=10):
    # ! Scaling is required for the evaluation of the InterHuman dataset.
    EMB_SCALE = 6 if dataname == 'InterHuman' else 1
    DIV_SCALE = 2 if dataname == 'InterHuman' else 1

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    for batch in val_loader:
        if dataname == 'InterHuman':
            if len(batch) == 5:
                name, text, motion1, motion2, motion_lens = batch
            else:
                name, text, motion1, motion2, _, _, motion_lens = batch

            # unnormalize motion for eval wrapper
            bs, seq, dim = motion1.shape
            et, em = eval_wrapper.get_co_embeddings([name, text, motion1, motion2, motion_lens])
        elif dataname == 'InterX':
            word_embeddings, pos_one_hots, caption, sent_lens, motions, _, _, m_lens, token = batch
            # unnormalize motion for eval wrapper
            bs, seq, dim = motions.shape
            et, em = eval_wrapper.get_co_embeddings(
                word_embs=word_embeddings,
                pos_ohot=pos_one_hots,
                cap_lens=sent_lens,
                motions=motions,
                m_lens=m_lens,
            )
        else:
            raise ValueError('Unknown dataset')

        motion_annotation_list.append(em)
        temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        R_precision_real += temp_R
        matching_score_real += temp_match

        nb_sample += bs

    for batch in val_loader:
        if dataname == 'InterHuman':
            if len(batch) == 5:
                name, text, motion1, motion2, motion_lens = batch
            else:
                name, text, motion1, motion2, _, _, motion_lens = batch

            # unnormalize motion for eval wrapper
            bs, seq, dim = motion1.shape
            et_pred, em_pred = eval_wrapper.get_co_embeddings([name, text, motion1, motion2, motion_lens])
        elif dataname == 'InterX':
            word_embeddings, pos_one_hots, caption, sent_lens, motions, _, _, m_lens, token = batch
            # unnormalize motion for eval wrapper
            bs, seq, dim = motions.shape
            et_pred, em_pred = eval_wrapper.get_co_embeddings(
                word_embs=word_embeddings,
                pos_ohot=pos_one_hots,
                cap_lens=sent_lens,
                motions=motions,
                m_lens=m_lens,
            )
        else:
            raise ValueError('Unknown dataset')

        motion_pred_list.append(em_pred)
        temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        R_precision += temp_R
        matching_score_pred += temp_match

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np, EMB_SCALE)
    mu, cov = calculate_activation_statistics(motion_pred_np, EMB_SCALE)

    diversity_real = calculate_diversity(motion_annotation_np, diversity_times if nb_sample > diversity_times else 100, EMB_SCALE, DIV_SCALE)
    diversity = calculate_diversity(motion_pred_np, diversity_times if nb_sample > diversity_times else 100, EMB_SCALE, DIV_SCALE)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Re {repeat_id} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    print(msg)
    return fid, diversity, R_precision, matching_score_pred


def predict_t2m_interx(motions, motion_lens, vq_model, trans, dataset, caption, m_tokens_len, time_steps, cond_scale):
    bs, mot_length, d = motions.shape
    pred_ids_lst = trans.generate(caption, m_tokens_len, time_steps, cond_scale, temperature=1)    
    m1_pred_ids, m2_pred_ids, dist_pred_ids = pred_ids_lst
    
    def reformat_token(token, num_seq):
        token_shape = token.shape
        token = token[..., :num_seq].reshape(token_shape[0], -1)
        return token

    pred_pose_eval = np.zeros_like(motions)
    for k in range(bs):
        s_seq = motion_lens[k]

        ######### [INFO] Eval by m_length
        sm1_ids = reformat_token(m1_pred_ids[k:k + 1], int(m_tokens_len[k].item()))
        sm2_ids = reformat_token(m2_pred_ids[k:k + 1], int(m_tokens_len[k].item()))
        sd_ids = reformat_token(dist_pred_ids[k:k + 1], int(m_tokens_len[k].item()))
        
        pred_pose1, pred_pose2 = vq_model.decode(sm1_ids, sm2_ids, sd_ids, s_seq)
        pred_pose1, pred_pose2 = pred_pose1.cpu().numpy(), pred_pose2.cpu().numpy()
        
        pred_pose1 = dataset.normalizer.backward(pred_pose1)
        pred_pose1 = pred_pose1.reshape(1, s_seq, 56, -1)
        
        pred_pose2 = dataset.normalizer.backward(pred_pose2)
        pred_pose2 = pred_pose2.reshape(1, s_seq, 56, -1)

        pred_pose = np.concatenate([pred_pose1, pred_pose2], axis=-1)
        pred_pose = pred_pose.reshape(1, s_seq, -1)
        pred_pose_eval[k:k + 1, :motion_lens[k], :] = pred_pose

    pred_pose_eval = torch.from_numpy(pred_pose_eval).cuda()
    return pred_pose_eval


def predict_t2m_inter_human(motions, motion_lens, vq_model, trans, dataset, text, m_tokens_len, time_steps, cond_scale, is_separate=True):
    assert len(m_tokens_len) == 96
    
    max_len = max(m_tokens_len)
    if is_separate:
        m1_pred_ids, m2_pred_ids, dist_pred_ids = [], [], []
        for i in range(3):
            pred_ids_lst = trans.generate(text[i*32:(i+1)*32], m_tokens_len[i*32:(i+1)*32], time_steps, cond_scale, max_len=max_len, temperature=1)
            m1_pred_ids.append(pred_ids_lst[0])
            m2_pred_ids.append(pred_ids_lst[1])
            dist_pred_ids.append(pred_ids_lst[2])
        m1_pred_ids = torch.cat(m1_pred_ids, dim=0)
        m2_pred_ids = torch.cat(m2_pred_ids, dim=0)
        dist_pred_ids = torch.cat(dist_pred_ids, dim=0)
    else:
        pred_ids_lst = trans.generate(text, m_tokens_len, time_steps, cond_scale, max_len=max_len, temperature=1)
        m1_pred_ids, m2_pred_ids, dist_pred_ids = pred_ids_lst
        
    
    def reformat_token(token, num_seq):
        token_shape = token.shape
        token = token[..., :num_seq].reshape(token_shape[0], -1)
        return token
    
    bs, mot_length, d = motions.shape
    pred_pose_eval1 = np.zeros_like(motions)
    pred_pose_eval2 = np.zeros_like(motions)
    for k in range(bs):
        ######### [INFO] Eval by m_length
        sm1_ids = reformat_token(m1_pred_ids[k:k + 1], int(m_tokens_len[k].item()))
        sm2_ids = reformat_token(m2_pred_ids[k:k + 1], int(m_tokens_len[k].item()))
        sd_ids = reformat_token(dist_pred_ids[k:k + 1], int(m_tokens_len[k].item()))
        
        # mot1_cidx, mot2_cidx, dist_cidx, mot_length
        pred_pose1, pred_pose2 = vq_model.decode(sm1_ids, sm2_ids, sd_ids, motion_lens[k])
        pred_pose1, pred_pose2 = pred_pose1.cpu().numpy(), pred_pose2.cpu().numpy()
        
        pred_pose1 = dataset.normalizer.backward(pred_pose1)
        pred_pose2 = dataset.normalizer.backward(pred_pose2)
        pred_pose_eval1[k:k + 1, :motion_lens[k], :] = pred_pose1
        pred_pose_eval2[k:k + 1, :motion_lens[k], :] = pred_pose2

    pred_pose_eval1 = torch.from_numpy(pred_pose_eval1).cuda()
    pred_pose_eval2 = torch.from_numpy(pred_pose_eval2).cuda()
    return pred_pose_eval1, pred_pose_eval2


@torch.no_grad()
def evaluation_transformer_mask(
    dataname,
    out_dir,
    val_loader,
    vq_model,
    trans,
    logger,
    writer,
    nb_iter,
    best_fid,
    best_iter,
    best_div,
    best_top1,
    best_top2,
    best_top3,
    best_matching,
    eval_wrapper,
    draw=True,
    save=True,
    cond_scale=None,
    diversity_times=300,
    mm_num_repeats=30,
    mm_num_times=10,
):

    def save_model(file_name):
        t2m_trans_state_dict = trans.state_dict()
        clip_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del t2m_trans_state_dict[e]
        state = {
            't2m_transformer': t2m_trans_state_dict,
            'nb_iter': nb_iter,
        }
        torch.save(state, file_name)

    # ! Scaling is required for the evaluation of the InterHuman dataset.
    EMB_SCALE = 6 if dataname == 'InterHuman' else 1
    DIV_SCALE = 2 if dataname == 'InterHuman' else 1

    trans.eval()
    vq_model.eval()
    dataset = val_loader.dataset

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    time_steps = 18
    if cond_scale is None:
        if dataname == "InterHuman":
            cond_scale = 2
        elif dataname == 'InterX':
            cond_scale = 2
        else:
            raise NotImplementedError

    nb_sample = 0
    for batch in tqdm(val_loader):
        if dataname == 'InterHuman':
            # torch.cuda.empty_cache()            
            name, text, motion1, motion2, motion_norm, joint_dist, motion_lens = batch
            m_tokens_len = (motion_lens // 4).cuda()
            bs = motion1.shape[0]

            # unnormalize motion for eval wrapper
            et, em = eval_wrapper.get_co_embeddings([name, text, motion1, motion2, motion_lens])
            
            # Either motion1 or motion2 can be input; either is fine—just to obtain the shape.
            pred_pose_eval1, pred_pose_eval2 = predict_t2m_inter_human(motion1, motion_lens, vq_model, trans, dataset, text, m_tokens_len, time_steps, cond_scale)
            et_pred, em_pred = eval_wrapper.get_co_embeddings([name, text, pred_pose_eval1, pred_pose_eval2, motion_lens])

        elif dataname == 'InterX':
            word_embeddings, pos_one_hots, caption, sent_lens, motions, motion_reset, joint_dist, motion_lens, token = batch
            m_tokens_len = (motion_lens // 4).cuda()

            # unnormalize motion for eval wrapper
            bs, seq, dim = motions.shape
            et, em = eval_wrapper.get_co_embeddings(
                word_embs=word_embeddings,
                pos_ohot=pos_one_hots,
                cap_lens=sent_lens,
                motions=motions,
                m_lens=motion_lens,
            )

            pred_pose_eval = predict_t2m_interx(motions, motion_lens, vq_model, trans, dataset, caption, m_tokens_len, time_steps, cond_scale)
            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_lens, pred_pose_eval, motion_lens)
            
        else:
            raise ValueError('Unknown dataset')

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np, EMB_SCALE)
    mu, cov = calculate_activation_statistics(motion_pred_np, EMB_SCALE)

    diversity_real = calculate_diversity(motion_annotation_np, diversity_times if nb_sample > diversity_times else 100, EMB_SCALE, DIV_SCALE)
    diversity = calculate_diversity(motion_pred_np, diversity_times if nb_sample > diversity_times else 100, EMB_SCALE, DIV_SCALE)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    multimodality = 0
    msg = f"--> \t Eva. Iter {nb_iter} :, \n\
                FID. {fid:.4f} , \n\
                Diversity Real. {diversity_real:.4f}, \n\
                Diversity. {diversity:.4f}, \n\
                R_precision_real. {R_precision_real}, \n\
                R_precision. {R_precision}, \n\
                matching_score_real. {matching_score_real}, \n\
                matching_score_pred. {matching_score_pred}, \n\
                multimodality. {multimodality:.4f}"

    logger.info(msg)

    if draw:
        writer.add_scalar('Test/FID', fid, nb_iter)
        writer.add_scalar('Test/Diversity', diversity, nb_iter)
        writer.add_scalar('Test/top1', R_precision[0], nb_iter)
        writer.add_scalar('Test/top2', R_precision[1], nb_iter)
        writer.add_scalar('Test/top3', R_precision[2], nb_iter)
        writer.add_scalar('Test/matching_score', matching_score_pred, nb_iter)
        writer.add_scalar('Test/multimodality', multimodality, nb_iter)

    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        logger.info(msg)
        best_fid, best_iter = fid, nb_iter
        if save:
            save_model(os.path.join(out_dir, 'net_best_fid.pth'))

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        logger.info(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        logger.info(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        logger.info(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        logger.info(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        logger.info(msg)
        best_top3 = R_precision[2]

    if save:
        save_model(os.path.join(out_dir, 'net_last.pth'))

    trans.train()
    vq_model.eval()
    return best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, multimodality, writer, logger


@torch.no_grad()
def evaluation_transformer_mask_test(
    dataname,
    repeat_id,
    val_loader,
    vq_model,
    trans,
    eval_wrapper,
    cond_scale,
    temperature,
    topkr,
    gsample=True,
    force_mask=False,
    cal_mm=True,
    diversity_times=300,
    mm_num_repeats=30,
    mm_num_times=10,
):
    # ! Scaling is required for the evaluation of the InterHuman dataset.
    EMB_SCALE = 6 if dataname == 'InterHuman' else 1
    DIV_SCALE = 2 if dataname == 'InterHuman' else 1

    trans.eval()
    vq_model.eval()
    dataset = val_loader.dataset

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    # 18 for training, 10 for testing
    time_steps = 10
    nb_sample = 0
    if cal_mm:
        num_mm_batch = 3
    else:
        num_mm_batch = 0

    for i, batch in enumerate(val_loader):
        if dataname == 'InterHuman':
            name, text, motion1, motion2, motion_norm, joint_dist, motion_lens = batch
            m_tokens_len = (motion_lens // 4).cuda()
            bs = motion1.shape[0]

            # unnormalize motion for eval wrapper
            et, em = eval_wrapper.get_co_embeddings([name, text, motion1, motion2, motion_lens])

            if i < num_mm_batch:
                motion_multimodality_batch = []
                for _ in range(mm_num_repeats):
                    pred_pose_eval1, pred_pose_eval2 = predict_t2m_inter_human(motion1, motion_lens, vq_model, trans, dataset, text, m_tokens_len, time_steps, cond_scale, is_separate=False)
                    
                    et_pred, em_pred = eval_wrapper.get_co_embeddings([name, text, pred_pose_eval1, pred_pose_eval2, motion_lens])
                    motion_multimodality_batch.append(em_pred.unsqueeze(1))
                motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1)  #(bs, 30, d)
                motion_multimodality.append(motion_multimodality_batch)
            else:
                pred_pose_eval1, pred_pose_eval2 = predict_t2m_inter_human(motion1, motion_lens, vq_model, trans, dataset, text, m_tokens_len, time_steps, cond_scale, is_separate=False)                
                et_pred, em_pred = eval_wrapper.get_co_embeddings([name, text, pred_pose_eval1, pred_pose_eval2, motion_lens])

        elif dataname == 'InterX':
            word_embeddings, pos_one_hots, caption, sent_lens, motions, motion_reset, joint_dist, motion_lens, token = batch
            m_tokens_len = (motion_lens // 4).cuda()
            bs = motions.shape[0]

            # unnormalize motion for eval wrapper
            et, em = eval_wrapper.get_co_embeddings(
                word_embs=word_embeddings,
                pos_ohot=pos_one_hots,
                cap_lens=sent_lens,
                motions=motions,
                m_lens=motion_lens,
            )

            if i < num_mm_batch:
                motion_multimodality_batch = []
                for _ in range(mm_num_repeats):
                    pred_pose_eval = predict_t2m_interx(motions, motion_lens, vq_model, trans, dataset, caption, m_tokens_len, time_steps, cond_scale)
                    et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_lens, pred_pose_eval, motion_lens)
                    motion_multimodality_batch.append(em_pred.unsqueeze(1))
                motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1)  #(bs, 30, d)
                motion_multimodality.append(motion_multimodality_batch)
            else:
                pred_pose_eval = predict_t2m_interx(motions, motion_lens, vq_model, trans, dataset, caption, m_tokens_len, time_steps, cond_scale)
                et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_lens, pred_pose_eval, motion_lens)
        else:
            raise ValueError('Unknown dataset')

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np, EMB_SCALE)
    mu, cov = calculate_activation_statistics(motion_pred_np, EMB_SCALE)

    diversity_real = calculate_diversity(motion_annotation_np, diversity_times if nb_sample > diversity_times else 100, EMB_SCALE, DIV_SCALE)
    diversity = calculate_diversity(motion_pred_np, diversity_times if nb_sample > diversity_times else 100, EMB_SCALE, DIV_SCALE)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    multimodality = 0
    if not force_mask and cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, mm_num_times)

    msg = f"--> \t Eva. Re {repeat_id} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, "
    msg += f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}, multimodality. {multimodality:.4f}"
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, multimodality


def predict_t2m_inter_human_reaction(motion_norm, joint_dist, motions, motion_lens, vq_model, trans, dataset, text, m_tokens_len, time_steps, cond_scale):
    assert len(m_tokens_len) == 96    
    nbp = 5
    pad_id = 8193
    max_len = max(m_tokens_len)
    
    bs = motion_norm.shape[0]
    m1_gt_ids = torch.ones((bs, nbp, max_len)).cuda().long() * pad_id  # [96, 5, 74]
    motion_norm = torch.tensor(motion_norm).cuda().float()
    joint_dist = torch.tensor(joint_dist).cuda().float()
    for k in range(bs):
        sm1_ids, sm2_ids, sd_ids = vq_model.encode(motion_norm[k:k+1, :motion_lens[k]], joint_dist[k:k+1, :motion_lens[k]])
        sm1_ids = sm1_ids.reshape(sm1_ids.shape[0], nbp, -1)  # [1, 5, token_len]        
        m1_gt_ids[k:k + 1, :, :sm1_ids.shape[-1]] = sm1_ids
    
    pred_ids_lst = trans.react(text, m1_gt_ids, m_tokens_len, time_steps, cond_scale, temperature=1)
    m1_pred_ids, m2_pred_ids, dist_pred_ids = pred_ids_lst        
    
    def reformat_token(token, num_seq):
        token_shape = token.shape
        token = token[..., :num_seq].reshape(token_shape[0], -1)
        return token
    
    bs, mot_length, d = motions.shape
    pred_pose_eval1 = np.zeros_like(motions)
    pred_pose_eval2 = np.zeros_like(motions)
    for k in range(bs):
        ######### [INFO] Eval by m_length
        sm1_ids = reformat_token(m1_pred_ids[k:k + 1], int(m_tokens_len[k].item()))
        sm2_ids = reformat_token(m2_pred_ids[k:k + 1], int(m_tokens_len[k].item()))
        sd_ids = reformat_token(dist_pred_ids[k:k + 1], int(m_tokens_len[k].item()))
        
        # mot1_cidx, mot2_cidx, dist_cidx, mot_length
        pred_pose1, pred_pose2 = vq_model.decode(sm1_ids, sm2_ids, sd_ids, motion_lens[k])
        pred_pose1, pred_pose2 = pred_pose1.cpu().numpy(), pred_pose2.cpu().numpy()
        
        pred_pose1 = dataset.normalizer.backward(pred_pose1)
        pred_pose2 = dataset.normalizer.backward(pred_pose2)
        pred_pose_eval1[k:k + 1, :motion_lens[k], :] = pred_pose1
        pred_pose_eval2[k:k + 1, :motion_lens[k], :] = pred_pose2

    pred_pose_eval1 = torch.from_numpy(pred_pose_eval1).cuda()
    pred_pose_eval2 = torch.from_numpy(pred_pose_eval2).cuda()
    return pred_pose_eval1, pred_pose_eval2


@torch.no_grad()
def evaluation_transformer_mask_reaction_test(
    dataname,
    repeat_id,
    val_loader,
    vq_model,
    trans,
    eval_wrapper,
    cond_scale,
    temperature,
    topkr,
    gsample=True,
    force_mask=False,
    cal_mm=True,
    diversity_times=300,
    mm_num_repeats=30,
    mm_num_times=10,
):
    # ! Scaling is required for the evaluation of the InterHuman dataset.
    EMB_SCALE = 6 if dataname == 'InterHuman' else 1
    DIV_SCALE = 2 if dataname == 'InterHuman' else 1

    trans.eval()
    vq_model.eval()
    dataset = val_loader.dataset

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    # 18 for training, 10 for testing
    time_steps = 10
    nb_sample = 0
    if cal_mm:
        num_mm_batch = 3
    else:
        num_mm_batch = 0

    for i, batch in enumerate(val_loader):
        if dataname == 'InterHuman':
            name, text, motion1, motion2, motion_norm, joint_dist, motion_lens = batch
            m_tokens_len = (motion_lens // 4).cuda()
            bs = motion1.shape[0]

            # unnormalize motion for eval wrapper
            et, em = eval_wrapper.get_co_embeddings([name, text, motion1, motion2, motion_lens])

            if i < num_mm_batch:
                motion_multimodality_batch = []
                for _ in range(mm_num_repeats):
                    pred_pose_eval1, pred_pose_eval2 = predict_t2m_inter_human_reaction(motion_norm, joint_dist, motion1, motion_lens, vq_model, trans, dataset, text, m_tokens_len, time_steps, cond_scale)
                    
                    et_pred, em_pred = eval_wrapper.get_co_embeddings([name, text, pred_pose_eval1, pred_pose_eval2, motion_lens])
                    motion_multimodality_batch.append(em_pred.unsqueeze(1))
                motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1)  #(bs, 30, d)
                motion_multimodality.append(motion_multimodality_batch)
            else:
                pred_pose_eval1, pred_pose_eval2 = predict_t2m_inter_human_reaction(motion_norm, joint_dist, motion1, motion_lens, vq_model, trans, dataset, text, m_tokens_len, time_steps, cond_scale)               
                et_pred, em_pred = eval_wrapper.get_co_embeddings([name, text, pred_pose_eval1, pred_pose_eval2, motion_lens])

        elif dataname == 'InterX':
            raise ValueError('Unknown dataset')
        else:
            raise ValueError('Unknown dataset')

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np, EMB_SCALE)
    mu, cov = calculate_activation_statistics(motion_pred_np, EMB_SCALE)

    diversity_real = calculate_diversity(motion_annotation_np, diversity_times if nb_sample > diversity_times else 100, EMB_SCALE, DIV_SCALE)
    diversity = calculate_diversity(motion_pred_np, diversity_times if nb_sample > diversity_times else 100, EMB_SCALE, DIV_SCALE)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample
    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    multimodality = 0
    if not force_mask and cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, mm_num_times)

    msg = f"--> \t Eva. Re {repeat_id} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, "
    msg += f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}, multimodality. {multimodality:.4f}"
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, multimodality


def predict_t2m_inter_human_inbetween(motion_norm, joint_dist, motions, motion_lens, vq_model, trans, dataset, text, m_tokens_len, time_steps, cond_scale):
    assert len(m_tokens_len) == 96    
    nbp = 5
    # self.mask_id = num_tokens, self.pad_id = num_tokens + 1
    mask_id = 8192
    pad_id = 8193
    ss = 0.1
    es = 0.9
    max_len = max(m_tokens_len)
    
    # Get the true token index of the first character
    bs = motion_norm.shape[0]
    m1_gt_ids = torch.ones((bs, nbp, max_len)).cuda().long() * pad_id  # [96, 5, 74]
    m2_gt_ids = torch.ones((bs, nbp, max_len)).cuda().long() * pad_id  # [96, 5, 74]
    sd_gt_ids = torch.ones((bs, nbp, max_len)).cuda().long() * pad_id  # [96, 5, 74]
    edit_mask = torch.zeros((bs, nbp, max_len)).cuda().bool()

    motion_norm = torch.tensor(motion_norm).cuda().float()
    joint_dist = torch.tensor(joint_dist).cuda().float()
    for k in range(bs):
        sm1_ids, sm2_ids, sd_ids = vq_model.encode(motion_norm[k:k+1, :motion_lens[k]], joint_dist[k:k+1, :motion_lens[k]])
        sm1_ids = sm1_ids.reshape(sm1_ids.shape[0], nbp, -1)  # [1, 5, token_len] 
        sm2_ids = sm2_ids.reshape(sm2_ids.shape[0], nbp, -1)  # [1, 5, token_len]   
        sd_ids = sd_ids.reshape(sd_ids.shape[0], nbp, -1)  # [1, 5, token_len]   
        
        start_idx = math.ceil(m_tokens_len[k] * ss)
        end_idx = math.floor(m_tokens_len[k] * es) - 1
        
        m1_gt_ids[k:k + 1, :, :sm1_ids.shape[-1]] = sm1_ids
        m1_gt_ids[k:k + 1, :, start_idx:end_idx] = mask_id
              
        m2_gt_ids[k:k + 1, :, :sm2_ids.shape[-1]] = sm2_ids
        m2_gt_ids[k:k + 1, :, start_idx:end_idx] = mask_id
           
        sd_gt_ids[k:k + 1, :, :sd_ids.shape[-1]] = sd_ids
        sd_gt_ids[k:k + 1, :, start_idx:end_idx] = mask_id
        
        edit_mask[k:k + 1, :, start_idx:end_idx] = True
    
    # Generate the second character's actions based on the text and the first character's actions
    pred_ids_lst = trans.edit(text, m1_gt_ids, m2_gt_ids, sd_gt_ids, edit_mask, m_tokens_len, time_steps, cond_scale, temperature=1)
    m1_pred_ids, m2_pred_ids, dist_pred_ids = pred_ids_lst        
    
    def reformat_token(token, num_seq):
        token_shape = token.shape
        token = token[..., :num_seq].reshape(token_shape[0], -1)
        return token
    
    bs, mot_length, d = motions.shape
    pred_pose_eval1 = np.zeros_like(motions)
    pred_pose_eval2 = np.zeros_like(motions)
    for k in range(bs):
        ######### [INFO] Eval by m_length
        sm1_ids = reformat_token(m1_pred_ids[k:k + 1], int(m_tokens_len[k].item()))
        sm2_ids = reformat_token(m2_pred_ids[k:k + 1], int(m_tokens_len[k].item()))
        sd_ids = reformat_token(dist_pred_ids[k:k + 1], int(m_tokens_len[k].item()))
        
        # mot1_cidx, mot2_cidx, dist_cidx, mot_length
        pred_pose1, pred_pose2 = vq_model.decode(sm1_ids, sm2_ids, sd_ids, motion_lens[k])
        pred_pose1, pred_pose2 = pred_pose1.cpu().numpy(), pred_pose2.cpu().numpy()
        
        pred_pose1 = dataset.normalizer.backward(pred_pose1)
        pred_pose2 = dataset.normalizer.backward(pred_pose2)
        pred_pose_eval1[k:k + 1, :motion_lens[k], :] = pred_pose1
        pred_pose_eval2[k:k + 1, :motion_lens[k], :] = pred_pose2

    pred_pose_eval1 = torch.from_numpy(pred_pose_eval1).cuda()
    pred_pose_eval2 = torch.from_numpy(pred_pose_eval2).cuda()
    return pred_pose_eval1, pred_pose_eval2


@torch.no_grad()
def evaluation_transformer_mask_inbetween_test(
    dataname,
    repeat_id,
    val_loader,
    vq_model,
    trans,
    eval_wrapper,
    cond_scale,
    temperature,
    topkr,
    gsample=True,
    force_mask=False,
    cal_mm=True,
    diversity_times=300,
    mm_num_repeats=30,
    mm_num_times=10,
):
    # ! Scaling is required for the evaluation of the InterHuman dataset.
    EMB_SCALE = 6 if dataname == 'InterHuman' else 1
    DIV_SCALE = 2 if dataname == 'InterHuman' else 1

    trans.eval()
    vq_model.eval()
    dataset = val_loader.dataset

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    # 18 for training, 10 for testing
    time_steps = 10
    nb_sample = 0
    if cal_mm:
        num_mm_batch = 3
    else:
        num_mm_batch = 0

    for i, batch in enumerate(val_loader):
        if dataname == 'InterHuman':
            name, text, motion1, motion2, motion_norm, joint_dist, motion_lens = batch
            m_tokens_len = (motion_lens // 4).cuda()
            bs = motion1.shape[0]

            # unnormalize motion for eval wrapper
            et, em = eval_wrapper.get_co_embeddings([name, text, motion1, motion2, motion_lens])

            if i < num_mm_batch:
                motion_multimodality_batch = []
                for _ in range(mm_num_repeats):
                    pred_pose_eval1, pred_pose_eval2 = predict_t2m_inter_human_inbetween(motion_norm, joint_dist, motion1, motion_lens, vq_model, trans, dataset, text, m_tokens_len, time_steps, cond_scale)
                    
                    et_pred, em_pred = eval_wrapper.get_co_embeddings([name, text, pred_pose_eval1, pred_pose_eval2, motion_lens])
                    motion_multimodality_batch.append(em_pred.unsqueeze(1))
                motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1)  #(bs, 30, d)
                motion_multimodality.append(motion_multimodality_batch)
            else:
                pred_pose_eval1, pred_pose_eval2 = predict_t2m_inter_human_inbetween(motion_norm, joint_dist, motion1, motion_lens, vq_model, trans, dataset, text, m_tokens_len, time_steps, cond_scale)               
                et_pred, em_pred = eval_wrapper.get_co_embeddings([name, text, pred_pose_eval1, pred_pose_eval2, motion_lens])

        elif dataname == 'InterX':
            raise ValueError('Unknown dataset')
        else:
            raise ValueError('Unknown dataset')

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np, EMB_SCALE)
    mu, cov = calculate_activation_statistics(motion_pred_np, EMB_SCALE)

    diversity_real = calculate_diversity(motion_annotation_np, diversity_times if nb_sample > diversity_times else 100, EMB_SCALE, DIV_SCALE)
    diversity = calculate_diversity(motion_pred_np, diversity_times if nb_sample > diversity_times else 100, EMB_SCALE, DIV_SCALE)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    multimodality = 0
    if not force_mask and cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, mm_num_times)

    msg = f"--> \t Eva. Re {repeat_id} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, "
    msg += f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}, multimodality. {multimodality:.4f}"
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, multimodality
