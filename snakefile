import msprime
import pandas as pd
import numpy as np
from scipy.stats import entropy
 
 
def run_simulation(
    ancestral_size, derived_size, bottleneck_size,
    split_time,
    a_sample_size, b_sample_size, sequence_length,
    mutation_rate):

    # Initialize the demography model
    demography = msprime.Demography()
    
    # Add an initial population
    demography.add_population(name="C", initial_size=ancestral_size)
    
    # Add source and recipient populations
    demography.add_population(name="A", initial_size=ancestral_size)
    demography.add_population(name="B", initial_size=derived_size)
    # Source and recipient population split at SPLIT_TIME in the past.
    demography.add_population_split(time=split_time, derived=["A","B"], ancestral="C")
    # Recipient population immediatately experiences a transmission bottleneck
    demography.add_simple_bottleneck(
        time=split_time-1, population="B", proportion=bottleneck_size)
    demography.sort_events()
    # Sample Source and Recipient populations at time 0 and set sample sizes
    # The sample size will be used to estimate allele frequencies in each population
    samples = [msprime.SampleSet(a_sample_size, population="A", time=0),      
               msprime.SampleSet(b_sample_size, population="B", time=0)]
    # Simulate ancestry with no recombination and haploid genomes.
    ts = msprime.sim_ancestry(
        samples=samples, # Sample N haploid genomes from each population
        demography=demography,
        ploidy=1,  # Set ploidy to 1 for haploid genomes
        sequence_length=sequence_length,  # Length of genome to simulate
        recombination_rate=0  # No recombination
    )

    # Simulate mutations
    mts = msprime.sim_mutations(ts, rate=mutation_rate)

    # Separate samples by population
    ancestral_samples = [s for s in mts.samples() if ts.node(s).population == ts.population(1).id]
    new_samples = [s for s in ts.samples() if ts.node(s).population == ts.population(2).id]
    vars = np.array([var.genotypes for var in mts.variants()]).T
    try:
        return {
            'A': vars[ancestral_samples],
            'B': vars[new_samples]
        }, mts
    except IndexError:
        return None, mts


def calculate_population_allele_freqs(genotypes):
    # calculate allele frequencies from sampled genotypes
    return np.array(genotypes, dtype=bool).sum(axis=0) / genotypes.shape[0]


def define_branches(a_genotype, b_genotype):
    """
    Determine which SNPs are on "A", "B", and "C" branch from two genotypes.
    """
    branches = []
    for a, b in zip(a_genotype, b_genotype):
        if a == 0 and b == 0:
            # Not a SNP
            branches.append(np.nan)
        elif a == 1 and b==1:
            branches.append("C")
        elif a == 1 and b==0:
            branches.append('A')
        elif a == 0 and b==1:
            branches.append('B')
        else:
            branches.append(np.nan)
    return branches
    

def segregating(val):
    """
    Define allele status as segregating fixed or not present in population.
    A segregating allele frequency is greater than 0 and less than 1, return 1.
    A fixed allele frequency is equal to 1, return 0.
    A zero frequency indicates a mutation that occurred after the split, return -1
    """
    if not val:
        return -1
    elif val == 1:
        return 0
    else:
        return 1


def mark_segregating_alleles(df):
    """
    Add columns for segregating status
    """
    df['Seg_A'] = df['Freq_A'].apply(segregating)
    df['Seg_B'] = df['Freq_B'].apply(segregating)
    return df

def calc_entropy(arr):
    """
    Calculate entropy for unique value counts
    """
    return entropy(arr, base=2)


def entropy_branch_result(a, b):
    """
    Return result of entropy test.
    """
    if a > b:
        return 1
    elif a == b:
        return 0
    elif a < b:
        return -1
    else:
        return 0


def method_entropy(df):
    """
    Calculate entropy for C->A branch and C->B branch for both populations.
    """
    # separate c->a branch and c->b branch, remove branches where one of them has zero frequency
    # as this represents independent lineages.
    a_c_branches = df[(df['Branches'].isin(['A', 'C'])) & (df['Seg_A'] != -1) & (df['Seg_B'] != -1)]
    b_c_branches = df[(df['Branches'].isin(['B', 'C'])) & (df['Seg_A'] != -1) & (df['Seg_B'] != -1)]
    a_a_entropy = calc_entropy(np.unique(a_c_branches['Freq_A'], return_counts=True)[-1])
    a_b_entropy = calc_entropy(np.unique(b_c_branches['Freq_A'], return_counts=True)[-1])
    b_a_entropy = calc_entropy(np.unique(a_c_branches['Freq_B'], return_counts=True)[-1])
    b_b_entropy = calc_entropy(np.unique(b_c_branches['Freq_B'], return_counts=True)[-1])
    a_branch_result = entropy_branch_result(a_a_entropy, b_a_entropy)
    b_branch_result = entropy_branch_result(a_b_entropy, b_b_entropy)
    result_sum = a_branch_result + b_branch_result
    # if both entropy values are greater in the source population
    # or one is greater and the other is equal, return correct
    if result_sum >= 1:
        return {
            'entropy': 'Correct'
        }
    # If one is correct and one incorrect, return ambiguous
    elif result_sum == 0:
        return {
            'entropy': 'Ambiguous'
        }
    # if both entropy values are greater in the recipient population
    # or one is greater and the other is equal, return wrong.
    elif result_sum < 0:
        return {
            'entropy': 'Wrong'
        }
    # Any other case, return unknown
    else:
        return {
            'entropy': 'Unknown'
        }
    

def method_phylo_range(df):
    """
    Evaluate the phylogenetic range
    """
    # Test if C-branch provides answer
    c_branch = df[df['Branches'] == "C"]

    a_on_c = c_branch[(c_branch['Seg_A'] == 1) & (c_branch['Seg_B'] != 1)].shape[0]
    b_on_c = c_branch[(c_branch['Seg_B'] == 1) & (c_branch['Seg_A'] != 1)].shape[0]
    # if the number of segregating snps on the C branch in the source population is
    # greater than the number of segregating snps on the C branch in the recipient 
    # population, return Correct.
    if a_on_c > b_on_c:
        return {
            'c_branch': 'Correct',
            'c_a_b_branch': 'Correct'
        }
    # If the number of segregating snps on the C branch in the source population is
    # less than the number of segregating snps on the C branch in the recipient 
    # population, return Wrong.
    elif a_on_c < b_on_c:
        return {
            'c_branch': 'Wrong',
            'c_a_b_branch': 'Wrong'
        }
    # If they are equal or there are no SNPs on the C branch. Continue to check A and B branches
    elif (len(c_branch) == 0) or (a_on_c == b_on_c):
        # Test a and b branches
        a_branch = df[df['Branches'] == "A"]
        b_branch = df[df['Branches'] == "B"]
        # If A and B have segregating alleles on both branches, ambiguous
        a_on_b = b_branch[(b_branch['Seg_A'] == 1)].shape[0]
        b_on_a = a_branch[(a_branch['Seg_B'] == 1)].shape[0]
        if a_on_b and b_on_a:
            return {
                'c_branch': "Ambiguous",
                'c_a_b_branch': 'Ambiguous'
            }
        # If no snps are segregating on branch B from source population and
        # none are segregating on branch A from recipient population, return Ambiguous. 
        elif not a_on_b and not b_on_a:
            return {
                'c_branch': "Ambiguous",
                'c_a_b_branch': "Ambiguous"
            }
        # If snps are segregating on branch B from the source population, return correct (ambiguous for C branch)
        elif a_on_b and not b_on_a:

            return {
                'c_branch': "Ambiguous",
                'c_a_b_branch': "Correct"
            }
        # If snps are segregating on branch A from the recipient population, return wrong (ambiguous on branch C)
        elif not a_on_b and b_on_a:
            return {
                'c_branch': "Ambiguous",
                'c_a_b_branch': "Wrong"
            }
        # all others return Unknown.
        else:
            return {
                'c_branch': 'Unknown',
                'c_a_b_branch': 'Unknown'
            }
        
    
def main(ancestral_size, derived_size, bottleneck_size,
    split_time,
    a_sample_size, b_sample_size, sequence_length,
    mutation_rate, reps, n_clones):
    """
    Run simulation and return results
    """
    data = []
    for rr in range(reps):
        # Run simulation
        sim,l = run_simulation(
            ancestral_size, derived_size, bottleneck_size,
            split_time,
            a_sample_size, b_sample_size, sequence_length,
            mutation_rate)
        if sim is None:
            continue
        # Calculate population frequencies
        a_freq = calculate_population_allele_freqs(sim['A'])
        b_freq = calculate_population_allele_freqs(sim['B'])
        # Randomly select clones from each
        a_clones = sim['A'][np.random.choice(len(sim['A']), n_clones, replace=False)]
        b_clones = sim['B'][np.random.choice(len(sim['B']), n_clones, replace=False)]
        # Pairwise compare clones in A and B
        pairwise_count = 0
        for a in a_clones:
            for b in b_clones:
                pairwise_count += 1
                table = pd.DataFrame.from_dict({
                    'Allele_A': a,
                    'Allele_B': b,
                    'Freq_A': a_freq,
                    'Freq_B': b_freq,
                    'Branches': define_branches(a, b)
                }).dropna().sort_values(['Branches', 'Freq_A'], ascending=False) # Drops non-snps from table
                table = mark_segregating_alleles(table)
                results = {**method_phylo_range(table), **method_entropy(table) }

                data.append([
                    rr, pairwise_count, ancestral_size, derived_size, bottleneck_size,
                    split_time, a_sample_size, b_sample_size, sequence_length,
                    mutation_rate, n_clones, reps,
                    results.get('c_branch', 'Unknown'), results.get('c_a_b_branch', 'Unknown'), results.get('entropy', 'Unknown')
                ])
    return pd.DataFrame(data, columns=[
        'Rep',
        'Pair',
        'Ancestral_size',
        'Derived_size',
        'Bottleneck_size',
        'Split_time',
        'A_sample_size',
        'B_sample_size',
        'Sequence_length',
        'Mutation_rate',
        'N_clones',
        'Reps',
        'PhyloRange-1',
        'PhyloRange-2',
        'Entropy'
    ])
    
            
ancestral_sizes = [1000, 5000, 50000, 500000]
derived_sizes = ancestral_sizes
bottleneck_sizes = [0.5, 0.75, 0.9, 0.95, 1]
split_times = [1000, 5000, 10000, 50000]
a_sample_sizes = [1000]
b_sample_sizes = a_sample_sizes
sequence_length = 2800000
mutation_rate = 1.6e-10

rule all:
    input:
        expand(
            "averages/{ancestral_size}_{derived_size}_{bottleneck_size}_{split_time}_{a_sample_size}_{b_sample_size}_{reps}_{n_clones}.tsv",
            ancestral_size=ancestral_sizes, derived_size=derived_sizes, bottleneck_size=bottleneck_sizes, split_time=split_times, a_sample_size=a_sample_sizes, b_sample_size=b_sample_sizes, reps=[1000], n_clones=[1]),
        expand(
            "averages/{ancestral_size}_{derived_size}_{bottleneck_size}_{split_time}_{a_sample_size}_{b_sample_size}_{reps}_{n_clones}.tsv",
            ancestral_size=ancestral_sizes, derived_size=derived_sizes, bottleneck_size=[0.75], split_time=split_times, a_sample_size=a_sample_sizes, b_sample_size=b_sample_sizes, reps=[1000], n_clones=[1, 2, 3, 4, 5]
        ),

            
rule run_simulation:
    """
    Run simulations and return raw results
    """
    output: "results/{ancestral_size}_{derived_size}_{bottleneck_size}_{split_time}_{a_sample_size}_{b_sample_size}_{reps}_{n_clones}.tsv"
    params:
        ancestral_size = lambda wc: int(wc.ancestral_size),
        derived_size = lambda wc: int(wc.derived_size),
        bottleneck_size = lambda wc: float(wc.bottleneck_size),
        split_time = lambda wc: int(wc.split_time),
        a_sample_size = lambda wc: int(wc.a_sample_size),
        b_sample_size = lambda wc: int(wc.b_sample_size),
        reps = lambda wc: int(wc.reps),
        n_clones = lambda wc: int(wc.n_clones),
        sequence_length = sequence_length,
        mutation_rate = mutation_rate

    run:
        df = main(params.ancestral_size, params.derived_size, params.bottleneck_size,
            params.split_time,
            params.a_sample_size, params.b_sample_size, params.sequence_length,
            params.mutation_rate, params.reps, params.n_clones)
        df.to_csv(output[0], index=False, sep="\t")


def get_combined_result(row):
    """
    Get a combined result using both methods
    """
    method1 = row['PhyloRange-2']
    method2 = row['Entropy']
    if method1 == 'Correct':
        if method2 == 'Correct' or method2 == 'Ambiguous':
            return 'Correct'
        if method2 == 'Wrong':
            return "Ambiguous"
    elif method1 == 'Ambiguous':
        if method2 == "Correct":
            return "Correct"
        if method2 == 'Ambiguous':
            return 'Ambiguous'
        if method2 == 'Wrong':
            return 'Wrong'
    elif method1 == 'Wrong':
        if method2 == 'Correct':
            return 'Ambiguous'
        if method2 == "Wrong" or method2 == 'Ambiguous':
            return "Wrong"
    else:
        return "Unknown"
        


rule summarize_simulation:
    """
    Combine results from multiple pairwise comparisons. Determine combined result from both methods.
    """
    input: "results/{ancestral_size}_{derived_size}_{bottleneck_size}_{split_time}_{a_sample_size}_{b_sample_size}_{reps}_{n_clones}.tsv"
    output: "summary/{ancestral_size}_{derived_size}_{bottleneck_size}_{split_time}_{a_sample_size}_{b_sample_size}_{reps}_{n_clones}.tsv"
    run:
        import pandas as pd 
        df = pd.read_csv(input[0], sep="\t")
        df["Combined"] = df[['PhyloRange-2', 'Entropy']].apply(get_combined_result, axis=1)
        data = []
        for n, d in df.groupby('Rep'):
            method_1 = d['PhyloRange-2'].value_counts().to_dict()
            method_2 = d['Entropy'].value_counts().to_dict()
            method_combo = d['Combined'].value_counts().to_dict()
            for method, res in {'PhyloRange': method_1, 'EntropyDiversity': method_2, "Combined": method_combo}.items():
                options = {'Correct': 0, 'Wrong': 0, 'Ambiguous': 0, 'Unknown': 0}
                if res.get('Correct', 0) > res.get('Wrong', 0):
                    # if there are more correct than wrong return correct
                    options['Correct'] = 1
                elif res.get('Correct', 0) == res.get('Wrong', 0):
                    # If there are an equal number of correct and wrong return ambiguous
                    options['Ambiguous'] = 1
                elif res.get('Correct', 0) < res.get('Wrong', 0):
                    # if there are fewer correct than wrong return wrong
                    options['Wrong'] = 1
                elif res.get('Correct', 0) == 0:
                    # if there are no correct then ambiguous
                    options['Ambiguous'] = 1
                else:
                    options['Unknown'] = 1
                for metric in ['Correct', 'Wrong', 'Ambiguous', 'Unknown']:
                    data.append([n, wildcards.ancestral_size, wildcards.derived_size, wildcards.bottleneck_size,
                                    wildcards.split_time, wildcards.a_sample_size, wildcards.b_sample_size, sequence_length,
                                    mutation_rate, wildcards.n_clones, wildcards.reps, metric, method, options.get(metric, 0)])
        data = pd.DataFrame(data, columns=[
            'Rep',
            'Ancestral_size',
            'Derived_size',
            'Bottleneck_size',
            'Split_time',
            'A_sample_size',
            'B_sample_size',
            'Sequence_length',
            'Mutation_rate',
            'N_clones',
            'Reps',
            'Metric',
            'Method',
            'Proportion'
        ])
        data.to_csv(output[0], sep="\t", index=False)


rule averages:
    """
    Calculate average proportion for each result for each method.
    """
    input: "summary/{ancestral_size}_{derived_size}_{bottleneck_size}_{split_time}_{a_sample_size}_{b_sample_size}_{reps}_{n_clones}.tsv"
    output: "averages/{ancestral_size}_{derived_size}_{bottleneck_size}_{split_time}_{a_sample_size}_{b_sample_size}_{reps}_{n_clones}.tsv"
    run:
        import pandas as pd
        df = pd.read_csv(input[0], sep="\t")
        method1 = df[df['Method'] == 'PhyloRange']
        method2 = df[df['Method'] == 'EntropyDiversity']
        method3 = df[df['Method'] == 'Combined']
        data = []
        for method, d in {'PhyloRange': method1, 'EntropyDiversity': method2, 'Combined': method3}.items():
            for metric in ['Correct', 'Wrong', 'Ambiguous', 'Unknown']:
                data.append([
                    wildcards.ancestral_size, wildcards.derived_size, wildcards.bottleneck_size,
                    wildcards.split_time, wildcards.a_sample_size, wildcards.b_sample_size, sequence_length,
                    mutation_rate, wildcards.n_clones, wildcards.reps, metric, method, d[d['Metric'] == metric]['Proportion'].mean()
                ])

        data = pd.DataFrame(data, columns=[
            'Ancestral_size',
            'Derived_size',
            'Bottleneck_size',
            'Split_time',
            'A_sample_size',
            'B_sample_size',
            'Sequence_length',
            'Mutation_rate',
            'N_clones',
            'Reps',
            'Metric',
            'Method',
            'AvgProportion'
        ])
        data.to_csv(output[0], sep="\t", index=False)

      


