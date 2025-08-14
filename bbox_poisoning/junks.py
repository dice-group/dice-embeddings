"""
high_scores,
mixed_scores,

high_gradients,
low_gradients,
mixed_gradients,

low_score_high_gradient,
high_score_high_gradient,
low_score_low_gradient,
high_score_low_gradient,

degree_high_score_high_triples,
degree_high_score_low_triples,
degree_low_score_high_triples,
degree_low_score_low_triples,

degree_high_grad_high_triples,
degree_high_grad_low_triples,
degree_low_grad_high_triples,
degree_low_grad_low_triples,

closeness_high_score_high_triples,
closeness_high_score_low_triples,
closeness_low_score_high_triples,
closeness_low_score_low_triples,

closeness_high_grad_high_triples,
closeness_high_grad_low_triples,
closeness_low_grad_high_triples,
closeness_low_grad_low_triples,

low_deg,
high_deg,
low_closeness,
"""

"""
        res_active_wbox_high_scores = store_poisoned_andeval(triples, high_scores, "high_scores", DB, top_k, corruption_type, experiment)
        res_wbox_high_scores.append(res_active_wbox_high_scores)

        res_active_wbox_mixed_scores = store_poisoned_andeval(triples, mixed_scores, "mixed_scores", DB, top_k, corruption_type, experiment)
        res_wbox_mixed_scores.append(res_active_wbox_mixed_scores)

        res_active_wbox_high_gradients = store_poisoned_andeval(triples, high_gradients, "high_gradients", DB, top_k, corruption_type, experiment)
        res_wbox_high_gradients.append(res_active_wbox_high_gradients)

        res_active_wbox_low_gradients = store_poisoned_andeval(triples, low_gradients, "low_gradients", DB, top_k, corruption_type, experiment)
        res_wbox_low_gradients.append(res_active_wbox_low_gradients)

        res_active_wbox_mixed_gradients = store_poisoned_andeval(triples, mixed_gradients, "mixed_gradients", DB, top_k, corruption_type, experiment)
        res_wbox_mixed_gradients.append(res_active_wbox_mixed_gradients)

        res_active_wbox_low_score_high_gradient = store_poisoned_andeval(triples, low_score_high_gradient, "low_score_high_gradient", DB, top_k, corruption_type, experiment)
        res_wbox_triples_with_low_score_high_gradien.append(res_active_wbox_low_score_high_gradient)

        res_active_wbox_high_score_high_gradient = store_poisoned_andeval(triples, high_score_high_gradient,
                                                                         "high_score_high_gradient", DB, top_k,
                                                                         corruption_type, experiment)
        res_wbox_triples_with_high_score_high_gradient.append(res_active_wbox_high_score_high_gradient)


        res_active_wbox_low_score_low_gradient = store_poisoned_andeval(triples, low_score_low_gradient,
                                                                          "low_score_low_gradient", DB, top_k,
                                                                          corruption_type, experiment)
        res_wbox_triples_with_low_score_low_gradient.append(res_active_wbox_low_score_low_gradient)

        res_active_wbox_high_score_low_gradient = store_poisoned_andeval(triples, high_score_low_gradient,
                                                                        "high_score_low_gradient", DB, top_k,
                                                                        corruption_type, experiment)
        res_wbox_triples_with_high_score_low_gradient.append(res_active_wbox_high_score_low_gradient)




        res_active_wbox_degree_high_score_high_triples = store_poisoned_andeval(triples, degree_high_score_high_triples,
                                                                                "degree_high_score_high_triples", DB, top_k,
                                                                                corruption_type, experiment)
        res_wbox_triples_with_degree_high_score_high_triples.append(res_active_wbox_degree_high_score_high_triples)

        res_active_wbox_degree_high_score_low_triples = store_poisoned_andeval(triples, degree_high_score_low_triples,
                                                                               "degree_high_score_low_triples", DB, top_k,
                                                                               corruption_type, experiment)
        res_wbox_triples_with_degree_high_score_low_triples.append(res_active_wbox_degree_high_score_low_triples)



        res_active_wbox_degree_low_score_high_triples = store_poisoned_andeval(triples, degree_low_score_high_triples,
                                                                               "degree_low_score_high_triples", DB, top_k,
                                                                               corruption_type, experiment)
        res_wbox_triples_with_degree_low_score_high_triples.append(res_active_wbox_degree_low_score_high_triples)



        res_active_wbox_degree_low_score_low_triples = store_poisoned_andeval(triples, degree_low_score_low_triples,
                                                                              "degree_low_score_low_triples", DB, top_k,
                                                                              corruption_type, experiment)
        res_wbox_triples_with_degree_low_score_low_triples.append(res_active_wbox_degree_low_score_low_triples)


        res_active_wbox_degree_high_grad_high_triples = store_poisoned_andeval(triples, degree_high_grad_high_triples,
                                                                               "degree_high_grad_high_triples", DB, top_k,
                                                                               corruption_type, experiment)
        res_wbox_triples_with_degree_high_grad_high_triples.append(res_active_wbox_degree_high_grad_high_triples)

        res_active_wbox_degree_high_grad_low_triples = store_poisoned_andeval(triples, degree_high_grad_low_triples,
                                                                              "degree_high_grad_low_triples", DB, top_k,
                                                                              corruption_type, experiment)
        res_wbox_triples_with_degree_high_grad_low_triples.append(res_active_wbox_degree_high_grad_low_triples)

        res_active_wbox_degree_low_grad_high_triples = store_poisoned_andeval(triples, degree_low_grad_high_triples,
                                                                              "degree_low_grad_high_triples", DB, top_k,
                                                                              corruption_type, experiment)
        res_wbox_triples_with_degree_low_grad_high_triples.append(res_active_wbox_degree_low_grad_high_triples)

        res_active_wbox_degree_low_grad_low_triples = store_poisoned_andeval(triples, degree_low_grad_low_triples,
                                                                             "degree_low_grad_low_triples", DB, top_k,
                                                                             corruption_type, experiment)
        res_wbox_triples_with_degree_low_grad_low_triples.append(res_active_wbox_degree_low_grad_low_triples)

        res_active_wbox_closeness_high_score_high_triples = store_poisoned_andeval(triples,
                                                                                   closeness_high_score_high_triples,
                                                                                   "closeness_high_score_high_triples", DB,
                                                                                   top_k, corruption_type, experiment)
        res_wbox_triples_with_closeness_high_score_high_triples.append(res_active_wbox_closeness_high_score_high_triples)



        res_active_wbox_closeness_high_score_low_triples = store_poisoned_andeval(triples, closeness_high_score_low_triples,
                                                                                  "closeness_high_score_low_triples", DB,
                                                                                  top_k, corruption_type, experiment)
        res_wbox_triples_with_closeness_high_score_low_triples.append(res_active_wbox_closeness_high_score_low_triples)

        res_active_wbox_closeness_low_score_high_triples = store_poisoned_andeval(triples, closeness_low_score_high_triples,
                                                                                  "closeness_low_score_high_triples", DB,
                                                                                  top_k, corruption_type, experiment)
        res_wbox_triples_with_closeness_low_score_high_triples.append(res_active_wbox_closeness_low_score_high_triples)

        res_active_wbox_closeness_low_score_low_triples = store_poisoned_andeval(triples, closeness_low_score_low_triples,
                                                                                 "closeness_low_score_low_triples", DB,
                                                                                 top_k, corruption_type, experiment)
        res_wbox_triples_with_closeness_low_score_low_triples.append(res_active_wbox_closeness_low_score_low_triples)

        res_active_wbox_closeness_high_grad_high_triples = store_poisoned_andeval(triples, closeness_high_grad_high_triples,
                                                                                  "closeness_high_grad_high_triples", DB,
                                                                                  top_k, corruption_type, experiment)
        res_wbox_triples_with_closeness_high_grad_high_triples.append(res_active_wbox_closeness_high_grad_high_triples)

        res_active_wbox_closeness_high_grad_low_triples = store_poisoned_andeval(triples, closeness_high_grad_low_triples,
                                                                                 "closeness_high_grad_low_triples", DB,
                                                                                 top_k, corruption_type, experiment)
        res_wbox_triples_with_closeness_high_grad_low_triples.append(res_active_wbox_closeness_high_grad_low_triples)

        res_active_wbox_closeness_low_grad_high_triples = store_poisoned_andeval(triples, closeness_low_grad_high_triples,
                                                                                 "closeness_low_grad_high_triples", DB,
                                                                                 top_k, corruption_type, experiment)
        res_wbox_triples_with_closeness_low_grad_high_triples.append(res_active_wbox_closeness_low_grad_high_triples)

        res_active_wbox_closeness_low_grad_low_triples = store_poisoned_andeval(triples, closeness_low_grad_low_triples,
                                                                                "closeness_low_grad_low_triples", DB, top_k,
                                                                                corruption_type, experiment)
        res_wbox_triples_with_closeness_low_grad_low_triples.append(res_active_wbox_closeness_low_grad_low_triples)
        """
#
# res_active_wbox_lowest_deg_triples = store_poisoned_andeval(triples, low_deg, "lowest_deg", DB, top_k,
#                                                            corruption_type, experiment)
# res_wbox_triples_with_lowest_deg_triples.append(res_active_wbox_lowest_deg_triples)

# res_active_wbox_highest_deg_triples = store_poisoned_andeval(triples, high_deg, "highest_deg", DB, top_k,
#                                                             corruption_type, experiment)
# res_wbox_triples_with_highest_deg_triples.append(res_active_wbox_highest_deg_triples)

# res_active_wbox_low_closeness = store_poisoned_andeval(triples, low_closeness, "low_closeness", DB, top_k,
#                                                            corruption_type, experiment)
# res_wbox_triples_with_lowest_closeness_triples.append(res_active_wbox_low_closeness)


# ("high_scores", res_wbox_high_scores),
("low_scores", res_wbox_low_scores),
# ("mixed_scores", res_wbox_mixed_scores),
# ("high_gradients", res_wbox_high_gradients),
# ("low_gradients", res_wbox_low_gradients),
# ("mixed_gradients", res_wbox_mixed_gradients),
# ("low_score_high_gradient", res_wbox_triples_with_low_score_high_gradien),
# ("high_score_high_gradient", res_wbox_triples_with_high_score_high_gradient),
# ("low_score_low_gradient", res_wbox_triples_with_low_score_low_gradient),
# ("high_score_low_gradient", res_wbox_triples_with_high_score_low_gradient),
("random", res_random),
# ("high_degree_high_score", res_wbox_triples_with_degree_high_score_high_triples),
# ("high_degree_low_score",  res_wbox_triples_with_degree_high_score_low_triples),
# ("low_degree_high_score",  res_wbox_triples_with_degree_low_score_high_triples),
# ("low_degree_low_score",   res_wbox_triples_with_degree_low_score_low_triples),
# ("high_degree_high_grad",  res_wbox_triples_with_degree_high_grad_high_triples),
# ("high_degree_low_grad",   res_wbox_triples_with_degree_high_grad_low_triples),
# ("low_degree_high_grad",   res_wbox_triples_with_degree_low_grad_high_triples),
# ("low_degree_low_grad",    res_wbox_triples_with_degree_low_grad_low_triples),
# ("high_closeness_high_score", res_wbox_triples_with_closeness_high_score_high_triples),
# ("high_closeness_low_score",  res_wbox_triples_with_closeness_high_score_low_triples),
# ("low_closeness_high_score",  res_wbox_triples_with_closeness_low_score_high_triples),
# ("low_closeness_low_score",   res_wbox_triples_with_closeness_low_score_low_triples),
# ("high_closeness_hig_gradh",  res_wbox_triples_with_closeness_high_grad_high_triples),
# ("high_closeness_low_grad",   res_wbox_triples_with_closeness_high_grad_low_triples),
# ("low_closeness_high_grad",   res_wbox_triples_with_closeness_low_grad_high_triples),
# ("low_closeness_low_grad",    res_wbox_triples_with_closeness_low_grad_low_triples),
# ("low_deg", res_wbox_triples_with_lowest_deg_triples),
# ("high_deg", res_wbox_triples_with_highest_deg_triples),
# ("low_closeness", res_wbox_triples_with_lowest_closeness_triples),


# "triple injection ratios": percentages,
# "high_scores": res_wbox_high_scores,
"low_scores": res_wbox_low_scores,
# "mixed_scores": res_wbox_mixed_scores,
# "high_gradients": res_wbox_high_gradients,
# "low_gradients": res_wbox_low_gradients,
# "mixed_gradients": res_wbox_mixed_gradients,
# "low_score_high_gradient": res_wbox_triples_with_low_score_high_gradien,
# "high_score_high_gradient": res_wbox_triples_with_high_score_high_gradient,
# "low_score_low_gradient": res_wbox_triples_with_low_score_low_gradient,
# "high_score_low_gradient": res_wbox_triples_with_high_score_low_gradient,
"random": res_random,
# "high_degree_high_score": res_wbox_triples_with_degree_high_score_high_triples,
# "high_degree_low_score":  res_wbox_triples_with_degree_high_score_low_triples,
# "low_degree_high_score":  res_wbox_triples_with_degree_low_score_high_triples,
# "low_degree_low_score":   res_wbox_triples_with_degree_low_score_low_triples,
# "high_degree_high_grad":  res_wbox_triples_with_degree_high_grad_high_triples,
# "high_degree_low_grad":   res_wbox_triples_with_degree_high_grad_low_triples,
# "low_degree_high_grad":   res_wbox_triples_with_degree_low_grad_high_triples,
# "low_degree_low_grad":    res_wbox_triples_with_degree_low_grad_low_triples,
# "high_closeness_high_score": res_wbox_triples_with_closeness_high_score_high_triples,
# "high_closeness_low_score":  res_wbox_triples_with_closeness_high_score_low_triples,
# "low_closeness_high_score":  res_wbox_triples_with_closeness_low_score_high_triples,
# "low_closeness_low_score":   res_wbox_triples_with_closeness_low_score_low_triples,
# "high_closeness_hig_gradh":  res_wbox_triples_with_closeness_high_grad_high_triples,
# "high_closeness_low_grad":   res_wbox_triples_with_closeness_high_grad_low_triples,
# "low_closeness_high_grad":   res_wbox_triples_with_closeness_low_grad_high_triples,
# "low_closeness_low_grad":    res_wbox_triples_with_closeness_low_grad_low_triples,
# "low_deg": res_wbox_triples_with_lowest_deg_triples,
# "high_deg": res_wbox_triples_with_highest_deg_triples,
# "low_closeness": res_wbox_triples_with_lowest_closeness_triples,
# "high_closeness": res_wbox_triples_with_high_closeness_triples


res_wbox_high_scores = []
res_wbox_low_scores = []
res_wbox_mixed_scores = []
res_wbox_high_gradients = []
res_wbox_low_gradients = []
res_wbox_mixed_gradients = []
res_wbox_triples_with_low_score_high_gradien = []
res_wbox_triples_with_high_score_high_gradient = []
res_wbox_triples_with_low_score_low_gradient = []
res_wbox_triples_with_high_score_low_gradient = []

res_wbox_triples_with_degree_high_score_high_triples = []
res_wbox_triples_with_degree_high_score_low_triples = []
res_wbox_triples_with_degree_low_score_high_triples = []
res_wbox_triples_with_degree_low_score_low_triples = []
res_wbox_triples_with_degree_high_grad_high_triples = []
res_wbox_triples_with_degree_high_grad_low_triples = []
res_wbox_triples_with_degree_low_grad_high_triples = []
res_wbox_triples_with_degree_low_grad_low_triples = []
res_wbox_triples_with_closeness_high_score_high_triples = []
res_wbox_triples_with_closeness_high_score_low_triples = []
res_wbox_triples_with_closeness_low_score_high_triples = []
res_wbox_triples_with_closeness_low_score_low_triples = []
res_wbox_triples_with_closeness_high_grad_high_triples = []
res_wbox_triples_with_closeness_high_grad_low_triples = []
res_wbox_triples_with_closeness_low_grad_high_triples = []
res_wbox_triples_with_closeness_low_grad_low_triples = []

res_wbox_triples_with_lowest_deg_triples = []
res_wbox_triples_with_highest_deg_triples = []
res_wbox_triples_with_lowest_closeness_triples = []
res_wbox_triples_with_high_closeness_triples = []

"""

    high_scores = sorted(adverserial_triples.copy(), key=lambda x: x[1], reverse=True)  # descending

    pairs = min(len(low_scores), len(high_scores))
    ordered_mix = []
    for i in range(pairs):
        ordered_mix.append(low_scores[i])
        ordered_mix.append(high_scores[i])

    if len(low_scores) > pairs:
        ordered_mix.extend(low_scores[pairs:])
    if len(high_scores) > pairs:
        ordered_mix.extend(high_scores[pairs:])

    mixed_scores = ordered_mix


    low_gradients = sorted(adverserial_triples.copy(), key=lambda x: x[2])  # ascending
    high_gradients = sorted(adverserial_triples.copy(), key=lambda x: x[2], reverse=True)  # descending

    pairs = min(len(low_gradients), len(high_gradients))
    ordered_mix_grad = []
    for i in range(pairs):
        ordered_mix_grad.append(low_gradients[i])
        ordered_mix_grad.append(high_gradients[i])

    if len(low_gradients) > pairs:
        ordered_mix_grad.extend(low_gradients[pairs:])
    if len(high_gradients) > pairs:
        ordered_mix_grad.extend(high_gradients[pairs:])

    mixed_gradients = ordered_mix_grad



    triples_with_low_score_high_gradient = select_low_score_high_gradient(adverserial_triples,
                                                                          score_percentile=50)

    triples_with_high_score_high_gradient = select_high_score_high_gradient(adverserial_triples, score_percentile=50)

    triples_with_low_score_low_gradient = select_low_score_low_gradient(adverserial_triples, score_percentile=50)

    triples_with_high_score_low_gradient = select_high_score_low_gradient(adverserial_triples, score_percentile=50)



    adverserial_triples_high_scores = [item[0] for item in high_scores]

    adverserial_triples_mixed_scores = [item[0] for item in mixed_scores]
    adverserial_triples_high_gradients = [item[0] for item in high_gradients]
    adverserial_triples_low_gradients = [item[0] for item in low_gradients]
    adverserial_triples_mixed_gradients = [item[0] for item in mixed_gradients]
    adverserial_triples_low_score_high_gradient = [item[0] for item in triples_with_low_score_high_gradient]
    adverserial_triples_high_score_high_gradient = [item[0] for item in triples_with_high_score_high_gradient]
    adverserial_triples_low_score_low_gradient = [item[0] for item in triples_with_low_score_low_gradient]
    adverserial_triples_high_score_low_gradient = [item[0] for item in triples_with_high_score_low_gradient]
    """

"""
   degree_high_score_high_triples = degree_high_score_high(corrupted_centerality, adverserial_triples)
   degree_high_score_low_triples = degree_high_score_low(corrupted_centerality, adverserial_triples)
   degree_low_score_high_triples = degree_low_score_high(corrupted_centerality, adverserial_triples)
   degree_low_score_low_triples = degree_low_score_low(corrupted_centerality, adverserial_triples)

   degree_high_grad_high_triples = degree_high_grad_high(corrupted_centerality, adverserial_triples)
   degree_high_grad_low_triples = degree_high_grad_low(corrupted_centerality, adverserial_triples)
   degree_low_grad_high_triples = degree_low_grad_high(corrupted_centerality, adverserial_triples)
   degree_low_grad_low_triples = degree_low_grad_low(corrupted_centerality, adverserial_triples)

   closeness_high_score_high_triples = closeness_high_score_high(corrupted_centerality, adverserial_triples)
   closeness_high_score_low_triples = closeness_high_score_low(corrupted_centerality, adverserial_triples)
   closeness_low_score_high_triples = closeness_low_score_high(corrupted_centerality, adverserial_triples)
   closeness_low_score_low_triples = closeness_low_score_low(corrupted_centerality, adverserial_triples)

   closeness_high_grad_high_triples = closeness_high_grad_high(corrupted_centerality, adverserial_triples)
   closeness_high_grad_low_triples = closeness_high_grad_low(corrupted_centerality, adverserial_triples)
   closeness_low_grad_high_triples = closeness_low_grad_high(corrupted_centerality, adverserial_triples)
   closeness_low_grad_low_triples = closeness_low_grad_low(corrupted_centerality, adverserial_triples)

   low_deg = low_degree(corrupted_centerality)
   high_deg = high_degree(corrupted_centerality)
   low_close = low_closeness(corrupted_centerality)
   high_close = high_closeness(corrupted_centerality)
   """

"""
            adverserial_triples_high_scores,

            adverserial_triples_mixed_scores,

            adverserial_triples_high_gradients,
            adverserial_triples_low_gradients,
            adverserial_triples_mixed_gradients,

            adverserial_triples_low_score_high_gradient,
            adverserial_triples_high_score_high_gradient,
            adverserial_triples_low_score_low_gradient,
            adverserial_triples_high_score_low_gradient,

            degree_high_score_high_triples,
            degree_high_score_low_triples,
            degree_low_score_high_triples,
            degree_low_score_low_triples,

            degree_high_grad_high_triples,
            degree_high_grad_low_triples,
            degree_low_grad_high_triples,
            degree_low_grad_low_triples,

            closeness_high_score_high_triples,
            closeness_high_score_low_triples,
            closeness_low_score_high_triples,
            closeness_low_score_low_triples,

            closeness_high_grad_high_triples,
            closeness_high_grad_low_triples,
            closeness_low_grad_high_triples,
            closeness_low_grad_low_triples,

            low_deg,
            high_deg,
            high_close,
            """
