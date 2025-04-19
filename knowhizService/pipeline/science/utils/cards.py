from typing import List, Optional, Set
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random
import logging

logger = logging.getLogger("kzpipeline.science.cards")

class CardsUtil:
    @staticmethod
    # Combine the definition and expansion of the flashcards
    def combine_cards(cards_def, cards_exp):
        combined_cards = {}
        for key in cards_def:
            combined_cards[key] = {
                "definition": cards_def[key],
                "expansion": cards_exp[key]
            }
        return combined_cards

    @staticmethod
    # Remove duplicate keywords based on cosine similarity
    def find_indices_to_remove(keywords: List[str], texts: Optional[List[List[str]]] = None, thresh: float = 0.8) -> Set[int]:
        """
        Identify indices of keywords to be removed based on the product of cosine similarities from multiple text contexts,
        or a single context if only one list of texts is provided.

        Parameters:
        - keywords (List[str]): A list of keywords for which similarity analysis is to be performed.
        - texts (Optional[List[List[str]]]): A list of lists, where each sublist contains texts used for vectorization in different contexts.
        If only one list is provided, it is used in a single context. If not provided, the 'keywords' list is used in a single context.
        - thresh (float): A threshold value for the product of similarities. Pairs with a product above this threshold are considered duplicates.

        Returns:
        - Set[int]: A set of indices of keywords to be removed.
        """
        if texts is None:
            texts = [keywords]

        try:
            # Compute cosine similarity matrices for each sublist in texts
            # logger.info(f"Computing cosine similarity matrices for {len(texts)} text contexts...")
            matrices = []
            # logger.info("\nLength of texts: ", len(texts))
            # logger.info("\nTextlist[0]: ", texts[0])
            # logger.info("\nTextlist[1]: ", texts[1])
            for text_list in texts:
                vectorizer = CountVectorizer().fit_transform(text_list)
                cosine_matrix = cosine_similarity(vectorizer)
                # logger.info(f"cosine_matrix: {cosine_matrix}, with text_list: {text_list}")
                matrices.append(cosine_matrix)
                # logger.info(f"matrices: {matrices}, with text_list: {text_list}")

            # Set to store indices of keywords to remove
            # logger.info(f"Finding indices to remove with a similarity threshold of {thresh}...")
            indices_to_remove = set()

            # Iterate through the upper triangular part of the matrices
            # logger.info(f"Comparing {len(keywords)} keywords...")
            for i in range(len(keywords)):
                for j in range(i + 1, len(keywords)):
                    # Calculate the product of similarity scores from each matrix
                    similarity_product = np.prod([matrices[k][i, j] for k in range(len(matrices))])
                    if similarity_product > thresh:
                        # Mark one of the keywords for removal
                        indices_to_remove.add(j)

            return indices_to_remove
        except Exception as e:
            logger.exception(f"An error occurred: {e}")
            return set()

    @staticmethod
    # work for one single main file and multiple supplementary files
    def divide_into_groups(n, ngroup=None, group_size=None):
        """
        Divides n items into a specified number of groups (ngroup) or based on a maximum group size (group_size).

        If only group_size is provided, the function calculates the minimum number of groups needed to accommodate
        all items without exceeding the specified group size. If any items are left over, an additional group is added.

        Parameters:
        - n (int): The total number of items to be divided into groups.
        - ngroup (int, optional): The number of groups to divide the items into. If provided, group_size is ignored.
        - group_size (int, optional): The maximum size of each group. Used to calculate ngroup if ngroup is not provided.

        Returns:
        - list[int]: A list where each element represents the size of a group, and the length of the list is the number of groups.
        """
        # If ngroup is not specified but group_size is, calculate the number of groups needed
        if ngroup is None and group_size is not None:
            # Calculate the base number of groups by dividing the total items by the group size
            ngroup = n // group_size
            # Calculate the remainder to see if an additional group is needed for leftover items
            remainder = n % group_size
            if remainder:
                ngroup += 1  # Add an extra group for any remaining items

        # Divide the total number of items by the number of groups to determine the base size of each group
        # and how many items are left to be evenly distributed among the groups
        quotient, extra = divmod(n, ngroup)

        # Generate and return a list of group sizes.
        # For the first 'extra' groups, add one additional item to each to distribute the remainder.
        return [quotient + 1 if i < extra else quotient for i in range(ngroup)]
    
    @staticmethod
    # locate the indices of a list of indices in their original lists
    def locate_indices_to_sets(indices, sets_size):
        """
        Locate the indices of a list of indices in their original lists. The current indices were obtained
        by combining original lists into one list.

        Parameters:
        - indices (List[int]): A list of indices from the combined list.
        - sets_size (List[int]): A list containing the sizes of the original lists.

        Returns:
        - List[Tuple[int, int]]: A list of tuples, where each tuple contains the original list index and the index within that list.
        """
        # Calculate the cumulative sizes to determine the boundaries of each original list
        cumulative_sizes = [0] + list(np.cumsum(sets_size))

        # Map each index in the combined list to its original list and index
        original_indices = []
        for idx in indices:
            for i in range(len(cumulative_sizes) - 1):
                if cumulative_sizes[i] <= idx < cumulative_sizes[i + 1]:
                    # Calculate the index within the original list
                    original_index = idx - cumulative_sizes[i]
                    original_indices.append((i, original_index))
                    break

        return original_indices

    @staticmethod
    def divide_corresponding_lists(keys_list, values_list, n):
        """
        # Example usage:
        keys_list0 = ['Ace', 'Two', 'Three', 'Four', 'Five']
        values_list0 = [1, 2, 3, 4, 5]
        n = 3
        keys_sublists, values_sublists = divide_corresponding_lists(keys_list0, values_list0, n)

        [['Two'], ['Four'], ['Three', 'Five', 'Ace']]
        [[2], [4], [3, 5, 1]]
        """
        # Pair the keys and values together
        paired_list = list(zip(keys_list, values_list))
        # Shuffle the paired list to maintain correspondence
        random.shuffle(paired_list)
        # Calculate the approximate size of each sublist
        avg_size = len(paired_list) // n
        if(avg_size >= 3):
            # Initialize the sublists
            keys_sublists = []
            values_sublists = []
            for i in range(n):
                start_index = i * avg_size
                # For the last sublist, take the remaining elements
                if i == n - 1:
                    sublist = paired_list[start_index:]
                else:
                    sublist = paired_list[start_index:start_index + avg_size]
                # Unzip the sublist into keys and values sublists
                keys, values = zip(*sublist) if sublist else ([], [])
                keys_sublists.append(list(keys))
                values_sublists.append(list(values))
        else:
            keys_sublists = [keys_list]
            values_sublists = [values_list]
        return keys_sublists, values_sublists