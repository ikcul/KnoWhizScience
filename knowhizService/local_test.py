import os
from multiprocessing import Pool
from pipeline.dev_tasks import generate_flashcards
import traceback
from functools import partial

def gen_config():
    return {
        'llm_source': 'openai',
        # 'llm_source': 'anthropic',
        'temperature': 0,
        "creative_temperature": 0.5,
        "openai_key_dir": ".env",
        "anthropic_key_dir": ".env",
        "results_dir": "pipeline/test_outputs/",
        "course_id_mapping_file": "pipeline/test_outputs/course_id_mapping.json",
        "max_test_multiple_choice_questions_per_section": 1,
        "max_test_short_answer_questions_per_section": 1,
        "quiz_random_seed": 5,
        "regions": ["Example"],    # Regions for flashcards expansion
        "flashcards_set_size": 30,
        "max_flashcards_size": 300,
        # "quality_check_size": 30,
        "rich_content": False,
        "options_list": ["Mindmap", "Table", "Formula", "Code"],
        "max_quiz_questions_per_section": 10,
        "definition_detail_level": 0,   # 0: no detail, 1: medium detail, 2: high detail
        "expansion_detail_level": 0,    # 0: no detail, 1: medium detail, 2: high detail
    }

def zero_shot_flashcards_para(course_description):
    para = {
        "zero_shot": True,
        "course_info": course_description,
        "keywords_per_chapter": 10,
    }
    # para.update(video_generation_params())  # Add video parameters
    para.update(gen_config())
    return para

def flashcards_para(main_filenames, supplementary_filenames=None):
    para = {
        "zero_shot": False,
        "book_dir": "pipeline/test_inputs/",
        "main_filenames": main_filenames,
        "supplementary_filenames": supplementary_filenames,
        "chunk_size": 2000,
        "similarity_score_thresh": 0.8,
        "num_context_pages": 50,
        "keywords_per_page": 1.5,
        "page_set_size": 5,
        "overlapping": 0,
        "link_flashcards_size": 30,
    }
    # para.update(video_generation_params())  # Add video parameters
    para.update(gen_config())
    return para

def local_test(params):
    try:
        zero_shot, course_description, main_files, supplementary_files = params
        if zero_shot:
            if course_description is None:
                raise ValueError("course_description cannot be None for zero-shot generation.")
            para = zero_shot_flashcards_para(course_description)
        else:
            if main_files is None:
                raise ValueError("main_files cannot be None for non-zero-shot generation.")
            para = flashcards_para(main_files, supplementary_files)
        generate_flashcards(para)
    except Exception as e:
        print(f"Error processing test case: {params}")
        print(f"Error details: {str(e)}")
        traceback.print_exc()

def process_test_cases(test_cases):
    for test_case in test_cases:
        try:
            local_test(test_case)
        except Exception as e:
            print(f"Failed to process test case: {test_case}")
            print(f"Error: {str(e)}")
            traceback.print_exc()
<<<<<<< HEAD

math_test_cases = [
    # (True, 'level:"Beginner",subject:"Mathematics",text:"Basic concepts of Algebra and solving linear equations"', None, None),
    # (True, 'level:"Intermediate",subject:"Mathematics",text:"Techniques of integration and their applications"', None, None),
    # (True, 'level:"Advanced",subject:"Mathematics",text:"Understand vector calculus and theorems like Stokes and Gauss"', None, None),
    # (True, 'level:"Beginner",subject:"Mathematics",text:"Introduction to Geometry: angles, triangles, and circles"', None, None),
    # (True, 'level:"Intermediate",subject:"Mathematics",text:"Probability and combinatorics fundamentals"', None, None),
    # (True, 'level:"Advanced",subject:"Mathematics",text:"Deep dive into real analysis: sequences, limits, and continuity"', None, None),
    # (True, 'level:"Beginner",subject:"Mathematics",text:"Understanding fractions, decimals, and percentages"', None, None),
    # (True, 'level:"Beginner",subject:"Mathematics",text:"Introduction to mathematical reasoning and proofs"', None, None),
    # (True, 'level:"Intermediate",subject:"Mathematics",text:"Matrix operations and basic linear algebra"', None, None),
    # (True, 'level:"Advanced",subject:"Mathematics",text:"Topology basics: open sets, continuity, and compactness"', None, None),
    # (True, 'level:"Intermediate",subject:"Mathematics",text:"Complex numbers and their geometric interpretation"', None, None),
    # (True, 'level:"Advanced",subject:"Mathematics",text:"Differential equations: methods and applications"', None, None),
    # (True, 'level:"Intermediate",subject:"Mathematics",text:"Graph theory fundamentals and applications"', None, None),
    # (True, 'level:"Advanced",subject:"Mathematics",text:"Abstract algebra: groups, rings, and fields"', None, None),
    # (True, 'level:"Intermediate",subject:"Mathematics",text:"Mathematical logic and set theory foundations"', None, None),
    # (True, 'level:"Advanced",subject:"Mathematics",text:"Explore number theory: divisibility, modular arithmetic, and prime numbers"', None, None),
    # (True, 'level:"Beginner",subject:"Mathematics",text:"Counting principles: permutations, combinations, and the basics of probability"', None, None),
    # (True, 'level:"Beginner",subject:"Mathematics",text:"Visualizing data: bar graphs, histograms, and pie charts"', None, None),
    # (True, 'level:"Intermediate",subject:"Mathematics",text:"Sequences and series: arithmetic and geometric progressions"', None, None),
    # (True, 'level:"Intermediate",subject:"Mathematics",text:"Solving systems of linear equations using matrices and row reduction"', None, None),
    # (True, 'level:"Intermediate",subject:"Mathematics",text:"Analytic geometry: conic sections and their properties"', None, None),
    # (True, 'level:"Intermediate",subject:"Mathematics",text:"Applications of derivatives: optimization and related rates"', None, None),
    # (True, 'level:"Advanced",subject:"Mathematics",text:"Measure theory and Lebesgue integration basics"', None, None),
    # (True, 'level:"Advanced",subject:"Mathematics",text:"Galois theory: solvability of equations and symmetry of roots"', None, None),
    # (True, 'level:"Advanced",subject:"Mathematics",text:"Mathematical modeling using differential equations and systems"', None, None),
    # (True, 'level:"Advanced",subject:"Mathematics",text:"Functional analysis: normed spaces and Banach spaces"', None, None),
    (True, 'level:"Beginner",subject:"Mathematics",text:"Introduction to statistics: mean, median, mode, and standard deviation"', None, None),
    (True, 'level:"Intermediate",subject:"Mathematics",text:"Multivariable calculus: partial derivatives and multiple integrals"', None, None),
    (True, 'level:"Advanced",subject:"Mathematics",text:"Category theory: functors, natural transformations, and adjunctions"', None, None),
    (True, 'level:"Beginner",subject:"Mathematics",text:"Basic number theory: prime factorization and divisibility rules"', None, None),
    (True, 'level:"Intermediate",subject:"Mathematics",text:"Numerical analysis: interpolation and numerical integration"', None, None),
    # (True, 'level:"Advanced",subject:"Mathematics",text:"Algebraic geometry: varieties and schemes"', None, None),
    # (True, 'level:"Beginner",subject:"Mathematics",text:"Introduction to trigonometry: sine, cosine, and tangent"', None, None),
    # (True, 'level:"Intermediate",subject:"Mathematics",text:"Discrete mathematics: logic, sets, and relations"', None, None),
    # (True, 'level:"Advanced",subject:"Mathematics",text:"Lie groups and Lie algebras: structure and representations"', None, None),
    # (True, 'level:"Beginner",subject:"Mathematics",text:"Basic probability: events, outcomes, and probability rules"', None, None),
    # (True, 'level:"Intermediate",subject:"Mathematics",text:"Fourier analysis: series and transforms"', None, None),
    # (True, 'level:"Advanced",subject:"Mathematics",text:"Algebraic topology: homotopy and homology theory"', None, None),
    # (True, 'level:"Beginner",subject:"Mathematics",text:"Introduction to coordinate geometry: points, lines, and planes"', None, None),
    # (True, 'level:"Intermediate",subject:"Mathematics",text:"Optimization theory: linear and nonlinear programming"', None, None),
    # (True, 'level:"Advanced",subject:"Mathematics",text:"Dynamical systems: stability and bifurcation theory"', None, None),
    # (True, 'level:"Beginner",subject:"Mathematics",text:"Basic set theory: operations and Venn diagrams"', None, None),
    # (True, 'level:"Intermediate",subject:"Mathematics",text:"Complex analysis: analytic functions and contour integration"', None, None),
    # (True, 'level:"Advanced",subject:"Mathematics",text:"Representation theory: characters and modules"', None, None),
    # (True, 'level:"Beginner",subject:"Mathematics",text:"Introduction to functions: domain, range, and transformations"', None, None),
    # (True, 'level:"Intermediate",subject:"Mathematics",text:"Differential geometry: curves and surfaces"', None, None),
    # (True, 'level:"Advanced",subject:"Mathematics",text:"Algebraic number theory: ideals and class groups"', None, None),
    # (True, 'level:"Beginner",subject:"Mathematics",text:"Basic algebra: polynomials and factoring"', None, None),
    # (True, 'level:"Intermediate",subject:"Mathematics",text:"Partial differential equations: classification and solutions"', None, None),
    # (True, 'level:"Advanced",subject:"Mathematics",text:"Operator theory: spectral theory and functional calculus"', None, None),
    # (True, 'level:"Beginner",subject:"Mathematics",text:"Introduction to mathematical logic: propositions, truth tables, and logical operators"', None, None),
    # (True, 'level:"Intermediate",subject:"Mathematics",text:"Game theory: Nash equilibrium and strategic decision making"', None, None),
    # (True, 'level:"Advanced",subject:"Mathematics",text:"Noncommutative geometry: quantum groups and spectral triples"', None, None),
    # (True, 'level:"Beginner",subject:"Mathematics",text:"Introduction to matrices: operations, determinants, and applications"', None, None),
    # (True, 'level:"Intermediate",subject:"Mathematics",text:"Cryptography: number theory applications and modern encryption methods"', None, None),
]
=======
>>>>>>> 53511019ab2598fb218fb418126295e55b6f55e7

if __name__ == "__main__":
    test_cases = math_test_cases
    # test_cases = [
    #     (True, """level:"Beginner",subject:"",text:"I want to learn Calculus""", None, None),
    #     # (True, """level:"pre-college",subject:"",text:"500 most frequent Chinese words""", None, None),
    #     #(True, """level:"Beginner",subject:"",text:"College Level Linear Algebra.""", None, None),
    #     #(True, "I want to learn Introduction to Law", None, None),
    #     #(True, "I want to learn Introduction to Biology", None, None),
    #     #(True, "I want to learn Introduction to History", None, None),
    #     #(True, "I want to learn Introduction to Physics", None, None),
    #     #(True, "I want to learn Introduction to Chemistry", None, None),
    #     #(True, "I want to learn Introduction to Politics", None, None),
    #     #(True, "I want to learn Introduction to Economics", None, None)
    #     # (True, "I want to learn Evidence Crash Course in Law School.", None, None),
    #     # (True, "I want to learn about How To Write Business Plan for Business School.", None, None),
    #     # (True, "I want to learn about Constitutional Law Crash Course for law school.", None, None),
    #     # (True, "I want to learn a Contracts Crash Course for business school.", None, None),
    #     # (True, "I want to learn about Property Crash Course for law school.", None, None),
    #     # (True, "I want to learn into level history of modern American art", None, None),
    #     # (True, "I want to learn beginner level financial accounting", None, None),
    #     # (True, "I want to learn itality of modern American art", None, None),
    #     # (True, "I want to learn advanced tax law for business school", None, None),
    #     # (True, "I want to learn EM physics...", None, None),
    #     # (True, "I want to learn Evidence Crash Course in Law School..", None, None),
    #     # (True, "I want to learn about the history of the United States test.?.", None, None),
    #     # (True, "I want to learn about How To Write Business Plan for Business School..", None, None),
    #     # (True, "I want to learn about Constitutional Law Crash Course for law school..", None, None),
    #     # (True, "I want to learn a Contracts Crash Course for business school..", None, None),
    #     # (True, "I want to learn about Property Crash Course for law school..", None, None),
    #     # (True, "I want to learn into level history of modern American art.", None, None),
    #     # (True, "I want to learn beginner level financial accounting.", None, None),
    #     # (True, "I want to learn itality of modern American art.", None, None),
    #     # (True, "I want to learn advanced tax law for business school.", None, None),
    #     # (False, None, ["https://arxiv.org/pdf/2405.10501"], ["https://www.youtube.com/watch?v=sOF0SsddQ_s", "https://en.wikipedia.org/wiki/Rabi_cycle"]),
    #     # (False, None, ["https://www.youtube.com/watch?v=sOF0SsddQ_s"], ["https://en.wikipedia.org/wiki/Generative_artificial_intelligence"]),
    #     # (False, None, ["https://en.wikipedia.org/wiki/Generative_model"], ["https://en.wikipedia.org/wiki/Generative_artificial_intelligence"]),
    #     # (False, None, ["https://www.youtube.com/watch?v=eBVi_sLaYsc"], []),
    #     # (False, None, ["https://en.wikipedia.org/wiki/Generative_model"], []),
    #     # (False, None, ["https://en.wikipedia.org/wiki/Web3"], []),
    #     # (False, None, ["12.apkg"], []),
    #     # (False, None, ["14.apkg"], []),
    #     # (False, None, ["0.pdf"], []),
    #     # (False, None, ["1.pdf"], []),
    #     # (False, None, ["2.pdf"], []),
    #     # (False, None, ["3.pdf"], []),
    #     # (False, None, ["NYSCourseGlossary.pdf"], []),
    #     # (False, None, ["5000_Most_Frequent_Chinese_Words_With_Wiktionary_Entries.apkg"], [])
    # ]

    try:
        # First try with multiprocessing
        with Pool(processes=os.cpu_count()) as pool:
            try:
                pool.map(local_test, test_cases)
            except Exception as e:
                print(f"Multiprocessing failed, falling back to sequential processing. Error: {str(e)}")
                process_test_cases(test_cases)
    except Exception as e:
        print(f"Failed to initialize multiprocessing pool. Running sequentially. Error: {str(e)}")
        process_test_cases(test_cases)
