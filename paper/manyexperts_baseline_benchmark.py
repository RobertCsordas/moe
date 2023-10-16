from run_tests import get_runs_and_infos

runs, infos = get_runs_and_infos(["wikitext103_small_match_manyexperts"])
print(infos[runs[0].id]["test_results"]["test/perplexity"])