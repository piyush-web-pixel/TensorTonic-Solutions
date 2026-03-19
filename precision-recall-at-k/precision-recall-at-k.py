def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    top_k=set(recommended[:k])
    relevant_set=set(relevant)
    Precision_k=top_k.intersection(relevant_set)
    count=0
    for i in Precision_k:
        count +=1
    p=count/k
    r=count/len(relevant)

    l=[p,r]

    return l
        