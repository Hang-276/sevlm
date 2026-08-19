import random
import os
import json
import re
from typing import List, Dict, Any, Tuple, Optional
import PIL.Image

class CLEVRSpotDiffGenerator:
    def __init__(self, images_dir: str = "/home/colligo/clevr-dataset-gen/output/replacement_images",
                 scenes_dir: str = "/home/colligo/clevr-dataset-gen/output/replacement_scenes",
                 num_players: int = 4, num_rounds: int = 2):
        """
        Initialize CLEVR spot-the-difference generator
        
        Args:
            images_dir: Directory containing comparison images
            scenes_dir: Directory containing scene JSON files
            num_players: Number of players in the game
            num_rounds: Number of clue rounds
        """
        self.images_dir = images_dir
        self.scenes_dir = scenes_dir
        self.num_players = num_players
        self.num_rounds = num_rounds
        
        # Load available comparison pairs
        self.comparison_pairs = self._load_comparison_pairs()
        
        # Role advantage baselines for adaptive reward shaping
        self.b_spy = 0.0      # EMA baseline of spy-role reward
        self.b_civ = 0.0      # EMA baseline of civilian-role reward
        self.alpha = 0.9      # EMA decay
        self.baseline_lock = None  # lock for multi-GPU sync (if ever needed)
        
        # stats for debugging
        self.baseline_update_count = 0
        self.total_spy_games = 0
        self.total_civ_games = 0
        
    def _load_comparison_pairs(self) -> List[Dict[str, str]]:
        """Load available comparison image pairs"""
        pairs = []
        if not os.path.exists(self.images_dir):
            print(f"Warning: Images directory {self.images_dir} does not exist")
            return pairs
            
        files = os.listdir(self.images_dir)
        original_files = [f for f in files if "_original.png" in f]
        
        for original_file in original_files:
            base_name = original_file.replace("_original.png", "")
            modified_file = f"{base_name}_modified.png"
            
            if modified_file in files:
                pairs.append({
                    "original": os.path.join(self.images_dir, original_file), 
                    "modified": os.path.join(self.images_dir, modified_file),
                    "base_name": base_name
                })
        
        return pairs
    
    def generate_game_data(self, epoch: int, sample_idx: int) -> Dict[str, Any]:
        """Generate a complete game data structure"""
        # Set seed for reproducibility
        random.seed(epoch * 10000 + sample_idx)
        
        if not self.comparison_pairs:
            # Fallback game data if no pairs available
            return self._create_fallback_game_data(epoch, sample_idx)
        
        # Select a random comparison pair
        pair = random.choice(self.comparison_pairs)
        
        # Randomly assign spy player (1-indexed)
        spy_player = random.randint(1, self.num_players)
        
        # Create player images: spy gets modified, others get original
        player_images = []
        for player_id in range(1, self.num_players + 1):
            if player_id == spy_player:
                player_images.append(pair["modified"])
            else:
                player_images.append(pair["original"])
        
        # Simplified game data - only focus on who is the spy
        game_data = {
            "game_id": f"clevr_epoch_{epoch}_sample_{sample_idx}",
            "epoch": epoch,
            "sample_idx": sample_idx,
            "num_players": self.num_players,
            "num_rounds": self.num_rounds,
            "spy_player": spy_player,
            "player_images": player_images,
            "comparison_data": {
                "original_image_path": pair["original"],
                "modified_image_path": pair["modified"],
                "base_name": pair["base_name"]
            }
        }
        
        return game_data


    def build_spy_game_for_base(
        self, base_name: str, spy_player: int, sample_idx: int = 0
    ) -> Optional[Dict[str, Any]]:
        """Build spot-diff game data for a SPECIFIC base scene + spy assignment.

        Unlike ``generate_game_data`` (which picks a random pair), this lets a
        policy-controlled caller choose the scene and spy slot deterministically
        while still using this generator's genuine spot-diff assembly (spy sees
        the modified image, civilians see the original). Returns None if the base
        scene is not in the discovered comparison pool.
        """
        pair = next(
            (p for p in self.comparison_pairs if p["base_name"] == base_name), None
        )
        if pair is None:
            return None
        spy_player = max(1, min(self.num_players, int(spy_player)))
        player_images = [
            pair["modified"] if pid == spy_player else pair["original"]
            for pid in range(1, self.num_players + 1)
        ]
        return {
            "game_id": f"clevr_spotdiff_{base_name}_spy{spy_player}_{sample_idx}",
            "sample_idx": sample_idx,
            "num_players": self.num_players,
            "num_rounds": self.num_rounds,
            "spy_player": spy_player,
            "player_images": player_images,
            "comparison_data": {
                "original_image_path": pair["original"],
                "modified_image_path": pair["modified"],
                "base_name": base_name,
            },
        }


    def _create_fallback_game_data(self, epoch: int, sample_idx: int) -> Dict[str, Any]:
        """Create fallback game data when no comparison pairs are available"""
        spy_player = random.randint(1, self.num_players)
        
        return {
            "game_id": f"clevr_epoch_{epoch}_sample_{sample_idx}_fallback",
            "epoch": epoch,
            "sample_idx": sample_idx,
            "num_players": self.num_players,
            "num_rounds": self.num_rounds,
            "spy_player": spy_player,
            "player_images": ["/dev/null"] * self.num_players,
            "comparison_data": {
                "original_image_path": "/dev/null",
                "modified_image_path": "/dev/null",
                "base_name": "fallback"
            }
        }
    
    def format_clue_phase_sample(self, game_data: Dict[str, Any], player_id: int, 
                                round_num: int, previous_clues_text: str, 
                                player_thinking_history: str = "") -> Dict[str, Any]:
        """Format a sample for the clue phase"""
        num_players = game_data["num_players"]
        num_rounds = game_data["num_rounds"]
        spy_player = game_data["spy_player"]
        
        # Determine if current player is spy or not
        if player_id == spy_player:
            role_info = "Your role: SPY (you have the different picture)"
            role_instruction = "• You are the spy with the different picture. Two objects in your picture are different from what others see, but they remain in the same positions as in the original image."
        else:
            role_info = "Your role: NOT SPY (you have the same picture as most others)"
            role_instruction = "• You are not the spy. Your picture is the same as most other players' pictures."
        
        # Create context about the game using the new prompt format
        game_context = f"""You are PLAYER {player_id} of the game "Who's the Odd One Out?". This is ROUND {round_num}. Your task is to describe your picture in one sentence, which can only be about the features of objects and the relationship between objects.

{role_info}

Game Instructions:
• There are {num_players} players in total. Only one player sees a different picture, and the pictures seen by all other players are the same.
• Player who are voted as the odd player will be severely punished, so you should convince others that you are not the odd player.
• In the odd player's picture, two objects are replaced with different ones, but they remain in the same positions as in the original image.  
{role_instruction}

What you should do:
1. Compare your picture with all previous clues to understand what others are seeing.
2. If you are NOT the spy: Describe a clear, specific feature of your picture that does not overlap with previous clues. This will help establish what the "normal" picture looks like.
3. If you are the spy: Be strategic! You know your picture is different, so either:
   - Describe features that are actually the same as others (if you can infer what they're seeing)
   - Describe features from your different picture in a way that doesn't reveal the differences
   - Focus on describing objects or aspects that might still be similar to what others see

Required output format:
First, conduct your private step-by-step reasoning — may include suspicions, probabilities, deception, etc.
Then, put your one-sentence description of your picture inside \\boxed{{}}, it needs to be different from the previous clue.
Example answer: \\boxed{{The cylinder is to the left of the cube}}; \\boxed{{The yellow object is the largest object}}; \\boxed{{The red object is the smallest object}}.
"""
        
        if previous_clues_text:
            game_context += f"\n\nPrevious think and clues (you can only can see think of yourself and clues of all players):\n\n{previous_clues_text}"
        else:
            # When there are no previous clues (usually round 1, player 1), add a helpful note
            game_context += f"\n\nPrevious think and clues (you can only can see think of yourself and clues of all players):\n (You are PLAYER {player_id} and is the first to speak，so there is no previous clue now.)"
        
        # Note: player_thinking_history is now integrated into previous_clues_text by the trainer
        # No need to add it separately here
        
        prompt = f"""<image>

{game_context}"""
        
        return {
            "image": [game_data["player_images"][player_id - 1]],
            "conversations": [
                {
                    "from": "human",
                    "value": prompt
                },
                {
                    "from": "gpt",
                    "value": "I need to look at my image and give a helpful clue. I can see several objects arranged in this scene.\n\\boxed{I can see some objects in the scene.}"
                }
            ]
        }
    
    def format_decision_phase_sample(self, game_data: Dict[str, Any], player_id: int,
                                   all_clues: str, player_thinking: str) -> Dict[str, Any]:
        """Format a sample for the decision phase for a specific player"""
        num_players = game_data["num_players"]
        spy_player = game_data["spy_player"]
        
        # Determine if current player is spy or not
        if player_id == spy_player:
            role_info = "Your role: SPY (you have the different picture)"
            role_instruction = "• You are the spy with the different picture. Two objects in your picture are different from what others see, but they remain in the same positions as in the original image."
        else:
            role_info = "Your role: NOT SPY (you have the same picture as most others)"
            role_instruction = "• You are not the spy. Your picture is the same as most other players' pictures."
        
        # Create context for decision making using the new prompt format
        decision_context = f"""You are PLAYER {player_id} of the game "Who's the Odd One Out?". You've just gone through the clue-giving stage, where each player provided a description of their picture. Now this is the decision-making stage. Your task is to vote on who is the Odd Player.

{role_info}

Game Instructions:
• There are {num_players} players in total. Only one player sees a different picture, and the pictures seen by all other players are the same.
• Player who are voted as the odd player will be severely punished. Therefore all players will try to convince others that they are not the odd player.
• In the odd player's picture, two objects are replaced with different ones, but they remain in the same positions as in the original image.  
{role_instruction}

What you should do:
1. Analyze all the previous clues to identify the spy player.
2. If you are NOT the spy: Look for clues that seem inconsistent with your picture or that seem deliberately vague/misleading. Vote for the player you believe is the spy. There is a reward for voting correctly.
3. If you are the spy: You need to vote for someone else to avoid suspicion. Choose another player to vote for, but DON'T vote for yourself as that would reveal your identity.

Required output format:
<think>
your private step-by-step reasoning — may include suspicions, probabilities, deception, etc.
</think>
<answer>
PLAYER_NUMBER or N/A
Example answer: 1; 2; 3; N/A. (Just number or N/A, don't contain any other text)
If you are uncertain about who is the spy, you can answer N/A. This will result in a smaller penalty than guessing incorrectly.
</answer>

All Clues in the Clue-giving Stage (you can see think of yourself and clues of all players):
{all_clues}"""

        # Note: player_thinking is now integrated into all_clues by the trainer
        # No need to add it separately here
        
        prompt = f"""<image>

{decision_context}"""
        
        # The correct answer depends on the player's perspective
        correct_spy = spy_player
        
        return {
            "image": [game_data["player_images"][player_id - 1]],
            "conversations": [
                {
                    "from": "human",
                    "value": prompt
                },
                {
                    "from": "gpt",
                    "value": f'Based on the clues, I need to identify the odd player. Let me analyze each clue carefully to find inconsistencies.\n\\boxed{{{correct_spy}}}'
                }
            ],
            "player_id": player_id,
            "correct_spy": correct_spy,
            "game_data": game_data
        }
    
    def format_god_decision_phase_sample(self, game_data: Dict[str, Any], all_clues: str) -> Dict[str, Any]:
        """Format a sample for the God's perspective decision phase (pure reasoning)"""
        num_players = game_data["num_players"]
        spy_player = game_data["spy_player"]
        
        # Create context for God's perspective decision making
        decision_context = f"""You are analyzing a game of "Who's the Odd One Out?" where {num_players} players have just provided clues about their pictures. Your task is to identify which player is the spy (the one with the different picture) based on their clues.

Game Setup:
• There are {num_players} players in total. Only one player sees a different picture (the spy), and the pictures seen by all other players are the same.
• In the spy's picture, two objects are replaced with different ones, but they remain in the same positions as in the original image.
• Each player provided a clue trying to describe their picture while avoiding suspicion.
• The spy tries to blend in by giving clues that seem consistent with what others see.
• Non-spy players try to give distinctive clues that help identify the spy.

Your task:
Analyze all the clues and determine which player is most likely the spy. Look for:
1. Clues that seem inconsistent with the majority
2. Clues that are deliberately vague or generic
3. Clues that might be trying to mislead others
4. Patterns that suggest deception or uncertainty

The image you see shows the ORIGINAL picture (what non-spy players see). Compare the clues against this reference.

Required output format:
First, conduct your private step-by-step reasoning — may include suspicions, probabilities, deception, etc.
Then, put your final answer (PLAYER_NUMBER or N/A) inside \\boxed{{}}. If you are uncertain, you can answer N/A.
Example answer: \\boxed{{1}}; \\boxed{{2}}; \\boxed{{3}}; \\boxed{{N/A}}. 

All Clues from the Clue-giving Stage:
{all_clues}"""
        
        prompt = f"""<image>

{decision_context}"""
        
        # The correct answer is always the spy player
        correct_spy = spy_player
        
        # Use original image path for God's perspective
        original_image_path = game_data["comparison_data"]["original_image_path"]
        
        return {
            "image": [original_image_path],
            "conversations": [
                {
                    "from": "human", 
                    "value": prompt
                },
                {
                    "from": "gpt",
                    "value": f'Based on analyzing all clues, I need to identify the spy player. Let me compare each clue against the original image to find discrepancies.\n\\boxed{{{correct_spy}}}'
                }
            ],
            "correct_spy": correct_spy,
            "game_data": game_data
        }
    
    def extract_thinking_from_clue(self, clue_response: str) -> str:
        """Extract thinking process from a clue response with flexible format support"""
        import re
        
        # First try old <think></think> format for backward compatibility
        think_match = re.search(r'<think>(.*?)</think>', clue_response, re.DOTALL)
        if think_match:
            return think_match.group(1).strip()
        
        # For new format: extract everything before \boxed{} as thinking
        boxed_match = re.search(r'\\\\?boxed\{.*?\}', clue_response, re.DOTALL)
        if boxed_match:
            # Extract everything before the boxed part as thinking
            thinking_part = clue_response[:boxed_match.start()].strip()
            # Clean up any prompt repetition or instructions that might be included
            if thinking_part:
                return thinking_part
        
        # If no clear structure, try to extract meaningful thinking content
        # Look for reasoning patterns in the text
        lines = clue_response.strip().split('\n')
        thinking_lines = []
        for line in lines:
            line = line.strip()
            # Skip empty lines and boxed content
            if line and not re.search(r'\\\\?boxed\{.*?\}', line):
                # Skip if it looks like a prompt instruction
                if not any(phrase in line.lower() for phrase in [
                    'you are player', 'your task is', 'game instructions', 
                    'required output format', 'example answer'
                ]):
                    thinking_lines.append(line)
        
        if thinking_lines:
            return '\n'.join(thinking_lines)
        
        return ""
    
    def extract_answer_from_clue(self, clue_response: str) -> str:
        """Extract answer content from a clue response with flexible format support"""
        import re
        
        # Try boxed format first (handle both single and double backslash)
        boxed_match = re.search(r'\\\\?boxed\{(.*?)\}', clue_response, re.DOTALL)
        if boxed_match:
            return boxed_match.group(1).strip()
        
        # Fallback: try old answer tags format for backward compatibility
        answer_match = re.search(r'<answer>(.*?)</answer>', clue_response, re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()
        
        # Try to find <answer> and extract everything after it
        answer_start_match = re.search(r'<answer>\s*(.*)', clue_response, re.DOTALL)
        if answer_start_match:
            answer_content = answer_start_match.group(1).strip()
            # Clean up any potential closing tags or extra content
            answer_content = re.split(r'</answer>|<think>|<answer>', answer_content)[0].strip()
            return answer_content
        
        # If no answer format found, return placeholder
        return "No valid clue provided."
    
    def extract_vote_from_decision(self, decision_response: str, debug_index: int = -1) -> Optional[Dict[str, Any]]:
        """Extract vote information from a decision response with flexible format support"""
        # Debug output for troubleshooting
        if debug_index >= 0 and debug_index < 3:
            print(f"[EXTRACT DEBUG {debug_index+1}] Input length: {len(decision_response)}")
            print(f"[EXTRACT DEBUG {debug_index+1}] Contains <answer>: {'<answer>' in decision_response}")
            print(f"[EXTRACT DEBUG {debug_index+1}] Contains </answer>: {'</answer>' in decision_response}")
            boxed_single = '\\boxed' in decision_response
            boxed_double = '\\\\boxed' in decision_response
            print(f"[EXTRACT DEBUG {debug_index+1}] Contains \\boxed: {boxed_single}")
            print(f"[EXTRACT DEBUG {debug_index+1}] Contains \\\\boxed: {boxed_double}")
        
        # Check if response has proper format with thinking content and boxed answer
        # For new format: extract thinking content before boxed answer
        boxed_match_for_thinking = re.search(r'\\\\?boxed\{.*?\}', decision_response, re.DOTALL)
        
        # Extract thinking content (everything before boxed answer for new format, or <think> tags for old format)
        think_match = re.search(r'<think>(.*?)</think>', decision_response, re.DOTALL)
        if not think_match and boxed_match_for_thinking:
            # New format: extract everything before boxed as thinking
            thinking_content = decision_response[:boxed_match_for_thinking.start()].strip()
            if thinking_content and len(thinking_content) > 10:
                think_match = True  # Mark as having thinking content
            else:
                think_match = False
        elif think_match:
            think_match = True
        else:
            think_match = False
        
        answer_content = None
        
        # Try boxed format first (handle both single and double backslash)
        boxed_match = re.search(r'\\\\?boxed\{(.*?)\}', decision_response, re.DOTALL)
        if boxed_match:
            answer_content = boxed_match.group(1).strip()
            if debug_index >= 0 and debug_index < 3:
                print(f"[EXTRACT DEBUG {debug_index+1}] Found boxed answer: '{answer_content}'")
        else:
            # Fallback: try old answer tags format for backward compatibility
            answer_match = re.search(r'<answer>(.*?)</answer>', decision_response, re.DOTALL)
            if answer_match:
                answer_content = answer_match.group(1).strip()
                if debug_index >= 0 and debug_index < 3:
                    print(f"[EXTRACT DEBUG {debug_index+1}] Found complete answer: '{answer_content}'")
            else:
                # Fallback: try to find <answer> and extract everything after it (like clue phase)
                answer_start_match = re.search(r'<answer>\s*(.*)', decision_response, re.DOTALL)
                if answer_start_match:
                    answer_content = answer_start_match.group(1).strip()
                    # Clean up any potential closing tags or extra content
                    answer_content = re.split(r'</answer>|<think>|<answer>', answer_content)[0].strip()
                    if debug_index >= 0 and debug_index < 3:
                        print(f"[EXTRACT DEBUG {debug_index+1}] Found partial answer: '{answer_content}'")
                else:
                    if debug_index >= 0 and debug_index < 3:
                        print(f"[EXTRACT DEBUG {debug_index+1}] No answer format found at all")
        

        
        # If we have answer content, try to extract vote
        if answer_content:
            # Check for N/A first
            answer_upper = answer_content.strip().upper()
            if answer_upper == "N/A" or answer_upper == "NA":
                return {
                    "voted_spy": "N/A",
                    "reasoning": "Uncertain vote"
                }
            
            # Extract just the number from answer content
            # Look for standalone numbers (could be just "3" or "Player 3")
            numbers = re.findall(r'\b([1-9])\b', answer_content)
            if numbers:
                return {
                    "voted_spy": int(numbers[0]),
                    "reasoning": "Direct number vote"
                }
            
            # Fallback: try to parse as JSON (old format compatibility)
            try:
                vote_data = json.loads(answer_content)
                if "spy" in vote_data:
                    return {
                        "voted_spy": int(vote_data["spy"]),
                        "reasoning": vote_data.get("reasoning", "")
                    }
            except:
                pass
        
        # If all else fails, return None (invalid vote)
        return None
    
    def calculate_game_rewards(self, game_data: Dict[str, Any], player_votes: List[Dict[str, Any]]) -> List[float]:
        """
        Calculate rewards for all players based on voting results
        Uses different reward schemes for spy vs non-spy players
        
        Args:
            game_data: Game data including spy player
            player_votes: List of vote dictionaries for each player
            
        Returns:
            List of rewards for each player
        """
        num_players = game_data["num_players"]
        spy_player = game_data["spy_player"]
        
        # Hyperparameters
        alpha = 0.6  # Being suspected penalty coefficient
        beta = 0.5   # Misleading bonus coefficient
        
        # Count votes for each player
        vote_counts = {i: 0 for i in range(1, num_players + 1)}
        
        valid_votes = []
        for player_id, vote_info in enumerate(player_votes, 1):
            if vote_info and "voted_spy" in vote_info:
                voted_spy = vote_info["voted_spy"]
                if voted_spy == "N/A":
                    valid_votes.append("N/A")  # N/A vote (doesn't count toward vote_counts)
                elif isinstance(voted_spy, int) and 1 <= voted_spy <= num_players:
                    vote_counts[voted_spy] += 1
                    valid_votes.append(voted_spy)
                else:
                    valid_votes.append(None)  # Invalid vote
            else:
                valid_votes.append(None)  # No vote
        
        rewards = [0.0] * num_players
        
        # Calculate rewards for each player
        for player_id in range(1, num_players + 1):
            player_idx = player_id - 1
            vote_info = player_votes[player_idx]
            
            if player_id == spy_player:
                # Spy player reward calculation
                reward = self._calculate_spy_reward(
                    player_id, vote_info, vote_counts, valid_votes, 
                    num_players, spy_player, alpha, beta
                )
            else:
                # Non-spy player reward calculation  
                reward = self._calculate_detective_reward(
                    player_id, vote_info, vote_counts, spy_player, 
                    num_players, alpha
                )
            
            # Clip reward to [-1, 1] range
            rewards[player_idx] = max(-1.0, min(1.0, reward))
        
        return rewards
    
    def _calculate_detective_reward(self, player_id: int, vote_info: Dict[str, Any], 
                                  vote_counts: Dict[int, int], spy_player: int, 
                                  num_players: int, alpha: float) -> float:
        """
        Calculate reward for non-spy players
        
        Components:
        1. Case-solving score: +1 if voted correctly, -1 if voted incorrectly
        2. Suspicion penalty: -α * recv[i] / (N-1)
        """
        # No valid vote -> severe penalty
        if not vote_info or "voted_spy" not in vote_info:
            return -1.0
        
        voted_spy = vote_info["voted_spy"]
        
        # Case-solving score: +1 for correct vote, -1 for incorrect vote
        if voted_spy == spy_player:
            case_solving_score = 1.0
        else:
            case_solving_score = -1.0
        
        # Suspicion penalty: being suspected is bad for detectives
        votes_received = vote_counts[player_id]
        suspicion_penalty = -alpha * votes_received / (num_players - 1)
        
        total_reward = case_solving_score + suspicion_penalty
        
        return total_reward
    
    def _calculate_spy_reward(self, spy_id: int, vote_info: Dict[str, Any], 
                            vote_counts: Dict[int, int], valid_votes: List[Optional[int]], 
                            num_players: int, spy_player: int, alpha: float, beta: float) -> float:
        """
        Calculate reward for spy player
        
        Components:
        1. Stealth score: +1 - recv[spy] / (N-1)
        2. Misleading bonus: +β * K_mislead / (N-1)
        3. Self-reveal penalty: -1 if spy votes for themselves
        """
        # Stealth score: fewer votes received = better
        votes_received = vote_counts[spy_id]
        stealth_score = 1.0 - votes_received / (num_players - 1)
        
        # Misleading bonus: count how many players voted for wrong targets (including N/A)
        mislead_count = 0
        for vote in valid_votes:
            if vote is not None and vote != spy_player:
                # Count both incorrect votes and N/A as "misled" (partial credit for N/A)
                if vote == "N/A":
                    mislead_count += 0.5  # Half credit for making players uncertain
                else:
                    mislead_count += 1.0  # Full credit for wrong votes
        
        misleading_bonus = beta * mislead_count / (num_players - 1)
        
        # Self-reveal penalty: spy voting for themselves is bad, N/A is neutral
        self_reveal_penalty = 0.0
        if vote_info and "voted_spy" in vote_info:
            voted_spy = vote_info["voted_spy"]
            if voted_spy == spy_player:
                self_reveal_penalty = -1.0  # Bad: self-reveal
            elif voted_spy == "N/A":
                self_reveal_penalty = 0.0   # Neutral: uncertainty is acceptable
        
        total_reward = stealth_score + misleading_bonus + self_reveal_penalty
        
        return total_reward

    def update_role_baselines(self, spy_reward: float, civilian_avg_reward: float, process_index: int = 0):
        """
        Update role-advantage baselines (EMA).
        
        Args:
            spy_reward: spy reward this round
            civilian_avg_reward: average civilian reward this round
            process_index: GPU process index (controls logging)
        """
        # EMA update
        old_b_spy = self.b_spy
        old_b_civ = self.b_civ
        
        self.b_spy = self.alpha * self.b_spy + (1 - self.alpha) * spy_reward
        self.b_civ = self.alpha * self.b_civ + (1 - self.alpha) * civilian_avg_reward
        
        # update stats
        self.baseline_update_count += 1
        self.total_spy_games += 1
        self.total_civ_games += 1
        
        # log on GPU-0 only (avoids duplicate logs)
        if process_index == 0 and self.baseline_update_count % 10 == 0:  # print every 10 updates
            print(f"[ROLE BASELINE] Update #{self.baseline_update_count}")
            print(f"  Spy baseline: {old_b_spy:.4f} → {self.b_spy:.4f} (game: {spy_reward:.4f})")
            print(f"  Civ baseline: {old_b_civ:.4f} → {self.b_civ:.4f} (game: {civilian_avg_reward:.4f})")
            print(f"  Total games: spy={self.total_spy_games}, civ={self.total_civ_games}")
    
    def get_role_advantages(self) -> tuple[float, float]:
        """
        Return the current role-advantage baselines.
        
        Returns:
            Tuple of (spy_baseline, civilian_baseline)
        """
        return self.b_spy, self.b_civ
    
    def apply_role_advantage_adjustment(self, rewards: list[float], spy_player: int, 
                                      process_index: int = 0) -> list[float]:
        """
        Apply role-advantage adjustment to rewards (single-GPU version, kept for compatibility).
        
        Args:
            rewards: raw reward list
            spy_player: spy player id (1-indexed)
            process_index: GPU process index
            
        Returns:
            adjusted reward list
        """
        adjusted_rewards = []
        spy_reward = rewards[spy_player - 1]  # Convert to 0-indexed
        civilian_rewards = [rewards[i] for i in range(len(rewards)) if i != spy_player - 1]
        civilian_avg = sum(civilian_rewards) / len(civilian_rewards) if civilian_rewards else 0.0
        
        # update baselines
        self.update_role_baselines(spy_reward, civilian_avg, process_index)
        
        # apply the advantage adjustment
        for i, reward in enumerate(rewards):
            player_id = i + 1
            if player_id == spy_player:
                # spy: subtract the spy baseline
                adjusted_reward = reward - self.b_spy
            else:
                # civilian: subtract the civilian baseline
                adjusted_reward = reward - self.b_civ
            adjusted_rewards.append(adjusted_reward)
        
        if process_index == 0:
            print(f"[ROLE ADVANTAGE] Applied advantage adjustment:")
            print(f"  Original rewards: {[f'{r:.3f}' for r in rewards]}")
            print(f"  Baselines: spy={self.b_spy:.3f}, civ={self.b_civ:.3f}")
            print(f"  Adjusted rewards: {[f'{r:.3f}' for r in adjusted_rewards]}")
            print(f"  Game stats: spy_r={spy_reward:.3f}, civ_avg={civilian_avg:.3f}")
        
        return adjusted_rewards
    
    def apply_unified_role_advantage_adjustment(self, rewards: list[float], spy_player: int,
                                              accelerator=None, process_index: int = 0) -> list[float]:
        """
        Apply role-advantage adjustment with a unified baseline (multi-GPU version).
        
        Args:
            rewards: raw reward list
            spy_player: spy player id (1-indexed)
            accelerator: Accelerator instance for cross-GPU communication
            process_index: GPU process index
            
        Returns:
            adjusted reward list
        """
        import torch
        
        spy_reward = rewards[spy_player - 1]  # Convert to 0-indexed
        civilian_rewards = [rewards[i] for i in range(len(rewards)) if i != spy_player - 1]
        civilian_avg = sum(civilian_rewards) / len(civilian_rewards) if civilian_rewards else 0.0
        
        if accelerator is not None:
            # gather spy and civilian rewards from all GPUs
            local_spy_reward = torch.tensor(spy_reward, device=accelerator.device)
            local_civ_reward = torch.tensor(civilian_avg, device=accelerator.device)
            
            # average across GPUs
            global_spy_reward = accelerator.reduce(local_spy_reward, reduction="mean")
            global_civ_reward = accelerator.reduce(local_civ_reward, reduction="mean")
            
            # update baselines from the global averages (all GPUs in sync)
            global_spy_reward_val = global_spy_reward.item()
            global_civ_reward_val = global_civ_reward.item()
            
            # unified baseline update
            self.update_unified_role_baselines(global_spy_reward_val, global_civ_reward_val, process_index)
            
            if process_index == 0:
                print(f"[UNIFIED BASELINE] Global averages across {accelerator.num_processes} GPUs:")
                print(f"  Local: spy_r={spy_reward:.4f}, civ_avg={civilian_avg:.4f}")
                print(f"  Global: spy_r={global_spy_reward_val:.4f}, civ_avg={global_civ_reward_val:.4f}")
        else:
            # no accelerator: fall back to single-GPU mode
            self.update_role_baselines(spy_reward, civilian_avg, process_index)
            if process_index == 0:
                print(f"[UNIFIED BASELINE] No accelerator provided, using single-GPU mode")
        
        # apply the advantage adjustment (unified baseline)
        adjusted_rewards = []
        for i, reward in enumerate(rewards):
            player_id = i + 1
            if player_id == spy_player:
                # spy: subtract the spy baseline
                adjusted_reward = reward - self.b_spy
            else:
                # civilian: subtract the civilian baseline
                adjusted_reward = reward - self.b_civ
            adjusted_rewards.append(adjusted_reward)
        
        if process_index == 0:
            print(f"[UNIFIED BASELINE] Applied unified advantage adjustment:")
            print(f"  Original rewards: {[f'{r:.3f}' for r in rewards]}")
            print(f"  Unified baselines: spy={self.b_spy:.4f}, civ={self.b_civ:.4f}")
            print(f"  Adjusted rewards: {[f'{r:.3f}' for r in adjusted_rewards]}")
        
        return adjusted_rewards
    
    def update_unified_role_baselines(self, global_spy_reward: float, global_civ_reward: float, process_index: int = 0):
        """
        Update role-advantage baselines from global averages (unified-baseline version).
        
        Args:
            global_spy_reward: global average spy reward across GPUs
            global_civ_reward: global average civilian reward across GPUs
            process_index: GPU process index (controls logging)
        """
        # EMA update
        old_b_spy = self.b_spy
        old_b_civ = self.b_civ
        
        self.b_spy = self.alpha * self.b_spy + (1 - self.alpha) * global_spy_reward
        self.b_civ = self.alpha * self.b_civ + (1 - self.alpha) * global_civ_reward
        
        # update stats
        self.baseline_update_count += 1
        self.total_spy_games += 1
        self.total_civ_games += 1
        
        # log on GPU-0 only (avoids duplicate logs)
        if process_index == 0 and self.baseline_update_count % 10 == 0:  # print every 10 updates
            print(f"[UNIFIED BASELINE] Update #{self.baseline_update_count}")
            print(f"  Spy baseline: {old_b_spy:.4f} → {self.b_spy:.4f} (global_game: {global_spy_reward:.4f})")
            print(f"  Civ baseline: {old_b_civ:.4f} → {self.b_civ:.4f} (global_game: {global_civ_reward:.4f})")
            print(f"  Total games: spy={self.total_spy_games}, civ={self.total_civ_games}")
            print(f"  Baseline synchronization: ALL GPUs use same unified baseline")
    
    def sync_baselines_across_gpus(self, accelerator, process_index: int = 0):
        """
        Sync baselines across GPUs (keeps them consistent).
        
        Args:
            accelerator: Accelerator instance
            process_index: GPU process index
        """
        import torch
        
        if accelerator is not None:
            # convert baselines to tensors and sync
            local_spy_baseline = torch.tensor(self.b_spy, device=accelerator.device)
            local_civ_baseline = torch.tensor(self.b_civ, device=accelerator.device)
            
            # average baselines across GPUs (keeps them identical everywhere)
            global_spy_baseline = accelerator.reduce(local_spy_baseline, reduction="mean")
            global_civ_baseline = accelerator.reduce(local_civ_baseline, reduction="mean")
            
            # set local baselines to the global averages
            self.b_spy = global_spy_baseline.item()
            self.b_civ = global_civ_baseline.item()
            
            if process_index == 0:
                print(f"[BASELINE SYNC] Synchronized baselines across {accelerator.num_processes} GPUs:")
                print(f"  Unified spy baseline: {self.b_spy:.4f}")
                print(f"  Unified civ baseline: {self.b_civ:.4f}")
        else:
            if process_index == 0:
                print(f"[BASELINE SYNC] No accelerator provided, skipping sync")


class CLEVRPotentialBasedRewardShaping:
    """
    Potential-Based Reward Shaping for CLEVR 'Spot-the-Difference' Game
    
    Two phases: clue / decision
    Potentials: Φ_vote (based on votes received), Φ_acc (based on accuracy)
    """
    
    def __init__(self, 
                 gamma: float = 0.99,           # Discount factor for potential shaping
                 lambda_fmt: float = 0.2,       # Format reward weight  
                 kappa_v: float = 0.5,          # Vote potential coefficient
                 beta_acc: float = 1.0):        # Accuracy reward weight
        self.gamma = gamma
        self.lambda_fmt = lambda_fmt  
        self.kappa_v = kappa_v
        self.beta_acc = beta_acc
    
    def phi_vote(self, votes: int, n_players: int) -> float:
        """
        Vote potential: Φ_vote = -κ_v * votes/(N-1)
        More votes received → lower potential (being suspected is bad)
        """
        return -self.kappa_v * votes / max(1, n_players - 1)
    
    def phi_acc(self, correct: bool) -> float:
        """
        Accuracy potential: Φ_acc = 1 if correct else 0
        """
        return 1.0 if correct else 0.0
    
    def clue_reward(self, fmt_ok: bool, votes_after: int, n_players: int) -> float:
        """
        Clue phase reward with potential-based shaping
        
        Args:
            fmt_ok: Whether format is correct
            votes_after: Number of votes received after this round
            n_players: Total number of players
        
        Returns:
            Shaped reward = format_reward + potential_shaping
        """
        # Format reward: +λ_fmt for good format, -λ_fmt for bad format  
        r_fmt = self.lambda_fmt * (2 * int(fmt_ok) - 1)
        
        # Potential-based shaping: γ * Φ_next - Φ_prev
        # Assume no votes before (Φ_prev = 0), votes_after determines Φ_next
        phi_prev = 0.0
        phi_next = self.phi_vote(votes_after, n_players)
        
        shaped_term = self.gamma * phi_next - phi_prev
        
        return r_fmt + shaped_term
    
    def decision_reward(self, fmt_ok: bool, correct: bool) -> float:
        """
        Decision phase reward with potential-based shaping
        
        Args:
            fmt_ok: Whether format is correct
            correct: Whether the vote/guess is correct
            
        Returns:
            Shaped reward = format_reward + accuracy_reward + potential_shaping
        """
        # Format reward
        r_fmt = self.lambda_fmt * (2 * int(fmt_ok) - 1)
        
        # Accuracy reward
        r_acc = (1.0 if correct else -1.0) * self.beta_acc
        
        # Potential-based shaping: γ * Φ_next - Φ_prev  
        # Φ_prev = 0 (no previous accuracy state), Φ_next = Φ_acc(correct)
        phi_prev = 0.0
        phi_next = self.phi_acc(correct)
        
        shaped_term = self.gamma * phi_next - phi_prev
        
        return r_fmt + r_acc + shaped_term


def create_clevr_spotdiff_data_generator(images_dir: str, scenes_dir: str, 
                                       num_players: int = 4, num_rounds: int = 2):
    """Create a CLEVR spot-the-difference data generator function for dynamic dataset"""
    generator = CLEVRSpotDiffGenerator(images_dir, scenes_dir, num_players, num_rounds)
    
    def data_generator(epoch: int, sample_idx: int) -> Dict[str, Any]:
        try:
            # Generate game data
            game_data = generator.generate_game_data(epoch, sample_idx)
            
            # Return game data for two-phase processing
            return {
                "game_data": game_data,
                "accu_reward_method": "clevr_spotdiff",
                "metadata": {
                    "epoch": epoch,
                    "sample_idx": sample_idx,
                    "game_type": "clevr_spotdiff",
                    "num_players": num_players,
                    "num_rounds": num_rounds
                }
            }
        except Exception as e:
            print(f"Error generating CLEVR sample {sample_idx}: {e}")
            # Return fallback
            fallback_game_data = generator._create_fallback_game_data(epoch, sample_idx)
            return {
                "game_data": fallback_game_data,
                "accu_reward_method": "clevr_spotdiff",
                "metadata": {
                    "epoch": epoch,
                    "sample_idx": sample_idx,
                    "game_type": "clevr_spotdiff_fallback",
                    "num_players": num_players,
                    "num_rounds": num_rounds
                }
            }
    
    # Create new reward functions for two-phase training
    def clevr_clue_format_reward_func(prompts, completions, **kwargs):
        """Enhanced format and content reward for clue phase - checks format, vocabulary, and content quality"""
        return _calculate_clevr_format_rewards(completions, phase="clue")
    
    def clevr_clue_format_with_votes_reward_func(prompts, completions, **kwargs):
        """Enhanced clue reward that considers both content quality AND voting results"""
        return _calculate_clue_rewards_with_vote_awareness(completions, kwargs)
    
    def clevr_clue_vote_penalty_reward_func(prompts, completions, **kwargs):
        """Vote penalty reward for clue phase (negative reward based on votes received)"""
        return _calculate_clevr_vote_penalty_rewards(generator, completions, kwargs, "clue")
    
    def clevr_decision_format_reward_func(prompts, completions, **kwargs):
        """Format reward for decision phase - checks for single number in answer"""
        return _calculate_clevr_format_rewards(completions, phase="decision")
    
    def clevr_decision_accuracy_reward_func(prompts, completions, **kwargs):
        """Accuracy reward for decision phase (whether player guessed correctly)"""
        return _calculate_clevr_accuracy_rewards(generator, completions, kwargs)
    
    # Keep the original reward functions for backward compatibility
    def spy_identification_reward_func(prompts, completions, **kwargs):
        """Legacy: Reward for correctly identifying the spy"""
        return _calculate_clevr_component_rewards(generator, completions, kwargs, "spy_identification")
    
    def vote_penalty_reward_func(prompts, completions, **kwargs):
        """Legacy: Penalty for receiving votes (being suspected)"""
        return _calculate_clevr_component_rewards(generator, completions, kwargs, "vote_penalty")
    
    def overall_game_reward_func(prompts, completions, **kwargs):
        """Legacy: Overall game performance reward"""
        return _calculate_clevr_component_rewards(generator, completions, kwargs, "overall")
    
    # Return both new and legacy reward functions
    return data_generator, (
        clevr_clue_format_reward_func, 
        clevr_clue_format_with_votes_reward_func,
        clevr_clue_vote_penalty_reward_func,
        clevr_decision_format_reward_func,
        clevr_decision_accuracy_reward_func,
        spy_identification_reward_func, 
        vote_penalty_reward_func, 
        overall_game_reward_func
    )


def _calculate_clevr_format_rewards(completions, phase="general"):
    """Calculate enhanced format rewards for CLEVR clue phase with content analysis"""
    import re
    
    if phase == "clue":
        return _calculate_enhanced_clevr_clue_rewards(completions)
    elif phase == "decision":
        return _calculate_decision_format_rewards(completions)
    else:
        # General format check (fallback)
        return _calculate_basic_format_rewards(completions)


def _calculate_enhanced_clevr_clue_rewards(completions):
    """Calculate enhanced rewards for CLEVR clue phase with vocabulary and content analysis"""
    import re
    from difflib import SequenceMatcher
    
    # CLEVR vocabulary categories
    CLEVR_COLORS = {
        'red', 'blue', 'green', 'yellow', 'purple', 'cyan', 'brown', 'gray', 'grey',
        'orange', 'pink', 'white', 'black', 'dark', 'light', 'bright'
    }
    
    CLEVR_SHAPES = {
        'cube', 'sphere', 'cylinder', 'ball', 'box', 'block', 'circle', 'round',
        'square', 'rectangular', 'cuboid', 'cylindrical', 'spherical', 'cubes',
        'spheres', 'cylinders', 'balls', 'boxes', 'blocks', 'circles', 'squares'
    }
    
    CLEVR_POSITIONS = {
        'left', 'right', 'front', 'behind', 'above', 'below', 'next', 'beside',
        'near', 'far', 'close', 'distant', 'center', 'middle', 'side', 'edge',
        'corner', 'top', 'bottom', 'between', 'among', 'around', 'inside', 'outside',
        'in front of', 'to the left', 'to the right', 'on top of', 'underneath'
    }
    
    CLEVR_SIZES = {
        'large', 'small', 'big', 'tiny', 'huge', 'massive', 'little', 'giant',
        'larger', 'smaller', 'bigger', 'smallest', 'largest', 'biggest', 'medium'
    }
    
    # Spatial relationship indicators (higher quality clues)
    CLEVR_RELATIONS = {
        'touching', 'aligned', 'parallel', 'perpendicular', 'diagonal', 'stacked',
        'grouped', 'scattered', 'clustered', 'arranged', 'positioned'
    }
    
    rewards = []
    
    # Extract all answer contents for repetition checking (flexible extraction)
    all_answers = []
    for comp in completions:
        # Try boxed format first (handle both single and double backslash)
        boxed_match = re.search(r'\\\\?boxed\{(.*?)\}', comp, re.DOTALL)
        if boxed_match:
            all_answers.append(boxed_match.group(1).strip().lower())
        else:
            # Fallback: try old answer tags format
            answer_match = re.search(r'<answer>(.*?)</answer>', comp, re.DOTALL)
            if answer_match:
                all_answers.append(answer_match.group(1).strip().lower())
            else:
                # Try to find <answer> and extract everything after it
                answer_start_match = re.search(r'<answer>\s*(.*)', comp, re.DOTALL)
                if answer_start_match:
                    answer_content = answer_start_match.group(1).strip()
                    # Clean up any potential closing tags or extra content
                    answer_content = re.split(r'</answer>|<think>|<answer>', answer_content)[0].strip()
                    all_answers.append(answer_content.lower())
                else:
                    all_answers.append("")
    
    for idx, completion in enumerate(completions):
        # Check basic format first
        think_match = re.search(r'<think>(.*?)</think>', completion, re.DOTALL)
        
        # Flexible answer extraction (handle both single and double backslash)
        boxed_match = re.search(r'\\\\?boxed\{(.*?)\}', completion, re.DOTALL)
        if boxed_match:
            answer_content = boxed_match.group(1).strip()
        else:
            # Fallback: try old answer tags format
            answer_match = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL)
            if answer_match:
                answer_content = answer_match.group(1).strip()
            else:
                # Try to find <answer> and extract everything after it
                answer_start_match = re.search(r'<answer>\s*(.*)', completion, re.DOTALL)
                if answer_start_match:
                    answer_content = answer_start_match.group(1).strip()
                    # Clean up any potential closing tags or extra content
                    answer_content = re.split(r'</answer>|<think>|<answer>', answer_content)[0].strip()
                else:
                    answer_content = ""
        
        if not (think_match and answer_content):
            rewards.append(0.0)  # No proper format
            continue
            
        think_content = think_match.group(1).strip()
        answer_lower = answer_content.lower()
        
        # Basic format requirements
        if len(think_content) < 10 or len(answer_content) == 0:
            rewards.append(0.1)  # Minimal format
            continue
        
        # Check if it's roughly one sentence (allow some flexibility)
        sentence_endings = len(re.findall(r'[.!?]+', answer_content))
        if sentence_endings > 2:  # Too many sentences
            rewards.append(0.2)
            continue
        
        # Base reward for proper format and single sentence
        reward = 0.2
        
        # === Content Analysis ===
        
        # Check for specific vocabulary categories
        color_words = [color for color in CLEVR_COLORS if color in answer_lower]
        shape_words = [shape for shape in CLEVR_SHAPES if shape in answer_lower]
        position_words = [pos for pos in CLEVR_POSITIONS if pos in answer_lower]
        size_words = [size for size in CLEVR_SIZES if size in answer_lower]
        relation_words = [rel for rel in CLEVR_RELATIONS if rel in answer_lower]
        
        # Reward for each category mentioned (encourage specific descriptions)
        if color_words:
            reward += 0.20  # Colors are very important for differentiation
        if shape_words:
            reward += 0.25  # Object types are crucial
        if position_words:
            reward += 0.20  # Spatial information is valuable
        if size_words:
            reward += 0.15  # Size comparisons help
        if relation_words:
            reward += 0.10  # Advanced spatial relationships
        
        # Bonus for rich, multi-dimensional descriptions
        categories_mentioned = sum([
            bool(color_words), bool(shape_words), bool(position_words), 
            bool(size_words), bool(relation_words)
        ])
        
        if categories_mentioned >= 2:
            reward += 0.1 * (categories_mentioned - 1)  # Bonus for combining multiple aspects
        
        # === Repetition Check ===
        
        # Check for repetition with previous clues (severe penalty)
        current_answer = answer_content.lower().strip()
        repetition_penalty = 0.0
        
        # Check similarity with all previous answers (before current index)
        for prev_idx in range(idx):
            prev_answer = all_answers[prev_idx]
            if prev_answer and current_answer:
                # Calculate similarity using sequence matcher
                similarity = SequenceMatcher(None, current_answer, prev_answer).ratio()
                
                # High similarity threshold - even moderate similarity gets penalized
                if similarity >= 0.7:  # Very similar
                    repetition_penalty += 0.8  # Severe penalty
                elif similarity >= 0.5:  # Moderately similar
                    repetition_penalty += 0.4  # Medium penalty
                elif similarity >= 0.3:  # Somewhat similar
                    repetition_penalty += 0.2  # Light penalty
                
                # Also check for exact word overlap (regardless of order)
                current_words = set(current_answer.split())
                prev_words = set(prev_answer.split())
                if len(current_words) > 0 and len(prev_words) > 0:
                    word_overlap = len(current_words & prev_words) / len(current_words | prev_words)
                    if word_overlap >= 0.6:  # 60% word overlap
                        repetition_penalty += 0.5
        
        # Apply repetition penalty
        reward -= repetition_penalty
        
        # === Quality Checks ===
        
        # SEVERE penalty for game information leakage (mentioning game mechanics)
        game_leak_words = {
            'odd', 'player', 'spy', 'different', 'same', 'vote', 'voting', 'suspect', 
            'suspicious', 'game', 'round', 'clue', 'guess', 'guessing', 'identify',
            'who is', 'which one', 'which player', 'other players', 'others',
            'i am', 'you are', 'we are', 'they are', 'everyone', 'somebody',
            'modified', 'original', 'changed', 'replaced', 'switch', 'switched'
        }
        
        game_leak_count = sum(1 for word in game_leak_words if word in answer_lower)
        if game_leak_count > 0:
            reward -= 0.5 * game_leak_count  # Severe penalty for game information leakage
        
        # Penalty for vague or generic language
        vague_phrases = {
            'i see', 'there is', 'there are', 'i notice', 'i can see', 'i think',
            'something', 'objects', 'things', 'stuff', 'picture shows', 'image has',
            'scene contains', 'various', 'some', 'several', 'many', 'few'
        }
        
        vague_count = sum(1 for phrase in vague_phrases if phrase in answer_lower)
        if vague_count > 0:
            reward -= 0.08 * vague_count
        
        # Penalty for being too generic or unhelpful
        if len(answer_content.split()) < 4:  # Too short
            reward -= 0.1
        elif len(answer_content.split()) > 20:  # Too long (not a single focused clue)
            reward -= 0.1
        
        # Bonus for specific comparative statements (very useful for the game)
        comparative_patterns = [
            r'\b(larger|smaller|bigger|taller|shorter)\s+than\b',
            r'\bto the (left|right|front|behind)\s+of\b',
            r'\b(next to|beside|near|far from)\b',
            r'\b(same|different)\s+(color|size|shape)\b'
        ]
        
        for pattern in comparative_patterns:
            if re.search(pattern, answer_lower):
                reward += 0.05  # Small bonus for comparative language
        
        # Ensure specific object identification rather than general descriptions
        if any(shape in answer_lower for shape in CLEVR_SHAPES) and any(color in answer_lower for color in CLEVR_COLORS):
            reward += 0.05  # Bonus for specific object identification (e.g., "red cube")
        
        # Final reward clamping and validation
        reward = max(0.0, min(1.0, reward))
        
        # Debug info (can be removed in production)
        if reward > 0.8:  # High quality clue
            debug_info = {
                'colors': color_words,
                'shapes': shape_words, 
                'positions': position_words,
                'sizes': size_words,
                'relations': relation_words,
                'categories': categories_mentioned,
                'answer': answer_content[:100]
            }
            # Could log this for debugging
        
        rewards.append(reward)
    
    return rewards


def _calculate_decision_format_rewards(completions):
    """Calculate format rewards specifically for decision phase"""
    import re
    
    rewards = []
    for completion in completions:
        # Check if completion has proper <think>...</think><answer>...</answer> format
        think_match = re.search(r'<think>(.*?)</think>', completion, re.DOTALL)
        
        # Flexible answer extraction for decision phase
        answer_match = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL)
        if answer_match:
            answer_content = answer_match.group(1).strip()
        else:
            # Try to find <answer> and extract everything after it
            answer_start_match = re.search(r'<answer>\s*(.*)', completion, re.DOTALL)
            if answer_start_match:
                answer_content = answer_start_match.group(1).strip()
                # Clean up any potential closing tags or extra content
                answer_content = re.split(r'</answer>|<think>|<answer>', answer_content)[0].strip()
            else:
                answer_content = ""
        
        if think_match and answer_content:
            think_content = think_match.group(1).strip()
            
            # Basic requirements: both sections need content
            if len(think_content) < 10 or len(answer_content) == 0:
                reward = 0.3  # Format exists but content is minimal
                rewards.append(reward)
                continue
            
            # For decision phase: answer should be exactly one number (1-9) or N/A
            answer_clean = answer_content.strip().upper()
            # Extract numbers from the answer
            numbers = re.findall(r'\b([1-9])\b', answer_clean)
            
            # Check for N/A option
            is_na = answer_clean == "N/A" or answer_clean == "NA"
            
            if len(numbers) == 1 and answer_clean.isdigit():
                reward = 1.0  # Perfect: exactly one digit, nothing else
            elif is_na:
                reward = 1.0  # Perfect: N/A is also valid
            elif len(numbers) == 1 and len(answer_clean.split()) <= 2:
                reward = 0.9  # Good: one number with minimal extra text
            elif len(numbers) == 1:
                reward = 0.7  # Acceptable: one number but with extra text
            else:
                reward = 0.3  # Wrong format: no number or multiple numbers
                
        elif think_match or answer_content:
            reward = 0.2  # Partial format (missing either think or answer)
        else:
            reward = 0.0  # No format
        
        rewards.append(reward)
    
    return rewards


def _calculate_basic_format_rewards(completions):
    """Basic format rewards for general use"""
    import re
    
    rewards = []
    for completion in completions:
        # For this basic format check, we'll create a simple temporary generator instance
        temp_generator = CLEVRSpotDiffGenerator()
        
        # Extract thinking and answer using the flexible extraction methods
        thinking_content = temp_generator.extract_thinking_from_clue(completion)
        boxed_match = re.search(r'\\\\?boxed\{(.*?)\}', completion, re.DOTALL)
        
        if boxed_match:
            answer_content = boxed_match.group(1).strip()
        else:
            # Fallback: try old answer tags format
            answer_match = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL)
            if answer_match:
                answer_content = answer_match.group(1).strip()
            else:
                # Try to find <answer> and extract everything after it
                answer_start_match = re.search(r'<answer>\s*(.*)', completion, re.DOTALL)
                if answer_start_match:
                    answer_content = answer_start_match.group(1).strip()
                    # Clean up any potential closing tags or extra content
                    answer_content = re.split(r'</answer>|<think>|<answer>', answer_content)[0].strip()
                else:
                    answer_content = ""
        
        # Check format using new criteria with decision-specific validation
        if thinking_content and answer_content:
            # For decision phase: answer should be a valid number (1-9) or N/A
            answer_clean = answer_content.strip().upper()
            # Extract numbers from the answer
            numbers = re.findall(r'\b([1-9])\b', answer_clean)
            
            # Check for N/A option
            is_na = answer_clean == "N/A" or answer_clean == "NA"
            
            if len(thinking_content) > 10:
                if len(numbers) == 1 and answer_clean.isdigit():
                    reward = 1.0  # Perfect: exactly one digit, nothing else
                elif is_na:
                    reward = 1.0  # Perfect: N/A is also valid
                elif len(numbers) == 1 and len(answer_clean.split()) <= 2:
                    reward = 0.9  # Good: one number with minimal extra text
                elif len(numbers) == 1:
                    reward = 0.7  # Acceptable: one number but with extra text
                else:
                    reward = 0.3  # Wrong format: no number or multiple numbers
            else:
                reward = 0.3  # Format exists but thinking is too short
        elif thinking_content or answer_content:
            reward = 0.2  # Partial format (missing either thinking or answer)
        else:
            reward = 0.0  # No format
        
        rewards.append(reward)
    
    return rewards


def _calculate_clue_rewards_with_vote_awareness(completions, kwargs):
    """Calculate clue rewards that consider both content quality and voting results"""
    try:
        # First calculate base content quality rewards
        content_rewards = _calculate_enhanced_clevr_clue_rewards(completions)
        
        # Check if vote data is available from the game results
        if 'game_vote_data' in kwargs:
            vote_data = kwargs['game_vote_data']
            samples = kwargs.get('samples', [])
            
            # Apply vote penalties to content rewards
            final_rewards = []
            for i, (content_reward, completion) in enumerate(zip(content_rewards, completions)):
                if i < len(samples):
                    player_id = samples[i].get('player_id', 1)
                    votes_received = vote_data.get(f'votes_for_player_{player_id}', 0)
                    total_voters = vote_data.get('total_voters', 4)
                    
                    # Apply same vote penalty as in enhanced two-phase system
                    vote_penalty = -0.6 * (votes_received / max(1, total_voters - 1))
                    final_reward = content_reward + vote_penalty
                else:
                    final_reward = content_reward
                
                final_rewards.append(final_reward)
            
            return final_rewards
        
        else:
            # If no vote data available, return content rewards only
            # This happens when vote data isn't available yet
            return content_rewards
            
    except Exception as e:
        print(f"Error in vote-aware clue calculation: {e}")
        # Fallback to content-only rewards
        return _calculate_enhanced_clevr_clue_rewards(completions)


def _calculate_clevr_vote_penalty_rewards(generator, completions, kwargs, phase):
    """Calculate vote penalty rewards for clue phase"""
    try:
        # This reward is calculated after all votes are collected
        # For now, return placeholder until full game completion
        # In a complete implementation, this would need access to the final voting results
        
        if 'game_vote_data' in kwargs:
            # If vote data is available, calculate actual penalties
            vote_data = kwargs['game_vote_data']
            rewards = []
            
            # Extract player info from completions/samples
            samples = kwargs.get('samples', [])
            for i, completion in enumerate(completions):
                if i < len(samples):
                    player_id = samples[i].get('player_id', 1)
                    # Calculate penalty based on votes received
                    votes_received = vote_data.get(f'votes_for_player_{player_id}', 0)
                    total_voters = vote_data.get('total_voters', 4)
                    
                    # Penalty: -0.5 * (votes_received / total_possible_votes)
                    penalty = -0.5 * (votes_received / max(1, total_voters - 1))
                    rewards.append(penalty)
                else:
                    rewards.append(0.0)
            
            return rewards
        else:
            # Return small random penalties as placeholder
            import random
            return [-0.1 * random.random() for _ in completions]
    
    except Exception as e:
        print(f"Error in vote penalty calculation: {e}")
        return [0.0] * len(completions)


def _calculate_clevr_accuracy_rewards(generator, completions, kwargs):
    """Calculate accuracy rewards for decision phase"""
    try:
        rewards = []
        samples = kwargs.get('samples', [])
        
        for i, completion in enumerate(completions):
            if i < len(samples):
                sample = samples[i]
                game_data = sample.get('game_data', {})
                spy_player = game_data.get('spy_player', 1)
                current_player_id = sample.get('player_id', 1)
                
                # Extract vote from completion
                vote_info = generator.extract_vote_from_decision(completion)
                if vote_info and "voted_spy" in vote_info:
                    voted_spy = vote_info["voted_spy"]
                    
                    # Different reward logic for spy vs non-spy players
                    if current_player_id == spy_player:
                        # Spy player: +0.5 if didn't vote for themselves, -1.0 if voted for themselves
                        if voted_spy == spy_player:
                            accuracy_reward = -1.0  # Spy voted for themselves (self-reveal)
                        elif voted_spy == "N/A":
                            accuracy_reward = 0.2   # Spy voted N/A (neutral strategy)
                        else:
                            accuracy_reward = 0.5   # Spy voted for someone else (good strategy)
                    else:
                        # Non-spy player: +1 for correct guess, -0.5 for N/A, -1 for incorrect
                        if voted_spy == spy_player:
                            accuracy_reward = 1.0   # Correct identification
                        elif voted_spy == "N/A":
                            accuracy_reward = -0.5  # Uncertain (better than wrong guess)
                        else:
                            accuracy_reward = -1.0  # Incorrect identification
                else:
                    accuracy_reward = -1.0  # No valid vote
                
                rewards.append(accuracy_reward)
            else:
                rewards.append(0.0)
        
        return rewards
    
    except Exception as e:
        print(f"Error in accuracy calculation: {e}")
        return [0.0] * len(completions)


def _calculate_vote_penalties_from_game_results(game_data, all_votes):
    """
    Calculate vote penalties for all players based on complete game voting results
    
    Args:
        game_data: Game data including player information
        all_votes: List of vote dictionaries from all players
        
    Returns:
        Dictionary mapping player_id to vote penalty
    """
    num_players = game_data["num_players"]
    
    # Count votes received by each player
    vote_counts = {player_id: 0 for player_id in range(1, num_players + 1)}
    valid_vote_count = 0
    
    for vote_info in all_votes:
        if vote_info and "voted_spy" in vote_info:
            voted_spy = vote_info["voted_spy"]
            if 1 <= voted_spy <= num_players:
                vote_counts[voted_spy] += 1
                valid_vote_count += 1
    
    # Calculate penalties: -0.5 * (votes_received / total_valid_votes)
    penalties = {}
    for player_id in range(1, num_players + 1):
        votes_received = vote_counts[player_id]
        if valid_vote_count > 0:
            penalty = -0.5 * (votes_received / valid_vote_count)
        else:
            penalty = 0.0
        penalties[player_id] = penalty
    
    return penalties


def _calculate_strategic_clue_rewards(game_data, all_votes, num_players, beta=0.1, lambda_param=0.1, 
                                    generator=None, apply_role_advantage=True):
    """
    Calculate strategic clue rewards based on voting game theory with zero-sum guarantee
    
    Formula:
    1. Count votes from non-spy players only
    2. Calculate suspicion potential: Ψ = v_U - \bar{v}_C
    3. Camp shared reward: 
       - Spy: Δ_U = -β * ΔΨ
       - Civilians: Δ_{C_j}^{share} = +β * ΔΨ / 3
    4. Individual suspicion reward for civilians: δ_{C_j} = -λ * (v_{C_j} - \bar{v}_C)
    5. Final rewards:
       - Spy: r_U = Δ_U
       - Civilian j: r_{C_j} = Δ_{C_j}^{share} + δ_{C_j}
    6. Zero-sum guarantee: r_U + Σr_{C_j} = 0
    7. Role advantage adjustment (if enabled):
       - r_spy_final = r_spy - b_spy
       - r_civ_final = r_civ - b_civ
    
    Args:
        game_data: Game data including spy player
        all_votes: List of vote dictionaries from decision phase
        num_players: Total number of players
        beta: Camp shared potential reward coefficient (≈0.05-0.2)
        lambda_param: Individual suspicion reward coefficient (≈0.05-0.15)
        generator: CLEVRSpotDiffGenerator instance for baseline tracking
        apply_role_advantage: Whether to apply role advantage adjustment
        
    Returns:
        Dict containing:
        - 'rewards': List of final rewards for each player
        - 'metrics': Dict with detailed metrics for wandb logging
    """
    spy_player = game_data["spy_player"]
    game_id = game_data.get("game_id", "unknown")
    
    # Get process index for GPU-specific logging (only show on GPU 0)
    # This function might be called from different contexts, so we handle the case where accelerator is not available
    try:
        import inspect
        frame = inspect.currentframe()
        while frame:
            if 'self' in frame.f_locals and hasattr(frame.f_locals['self'], 'accelerator'):
                process_index = frame.f_locals['self'].accelerator.process_index
                break
            frame = frame.f_back
        else:
            process_index = 0  # Default to 0 if not found
    except:
        process_index = 0  # Fallback
    
    if process_index == 0:
        print(f"[STRATEGIC CLUE] Game {game_id}: Calculating rewards for spy=Player {spy_player}")
    
    # Count votes from all God decisions (not player-specific votes)
    vote_counts = {player_id: 0 for player_id in range(1, num_players + 1)}
    total_god_voters = 0
    na_votes = 0
    invalid_votes = 0
    
    # Log all God decision votes first
    if process_index == 0:
        print(f"[STRATEGIC CLUE] All God decision votes:")
    for god_idx, vote_info in enumerate(all_votes):
        if vote_info and "voted_spy" in vote_info:
            voted_spy = vote_info["voted_spy"]
            if process_index == 0:
                print(f"  God decision {god_idx + 1}: voted for {voted_spy}")
        else:
            if process_index == 0:
                print(f"  God decision {god_idx + 1}: invalid vote [no vote_info or voted_spy]")
    
    # Count valid votes from all God decisions
    for god_idx, vote_info in enumerate(all_votes):
        # Count ALL God decisions (no filtering based on spy/civilian)
        if vote_info and "voted_spy" in vote_info:
            voted_spy = vote_info["voted_spy"]
            
            if voted_spy == "N/A" or voted_spy == "NA":
                # N/A votes don't count toward anyone's vote count
                na_votes += 1
                if process_index == 0:
                    print(f"  God decision {god_idx + 1} voted N/A - not counted toward any player")
            elif isinstance(voted_spy, int) and 1 <= voted_spy <= num_players:
                vote_counts[voted_spy] += 1
                total_god_voters += 1
                if process_index == 0:
                    print(f"  God decision {god_idx + 1} voted for Player {voted_spy} - COUNTED")
            else:
                invalid_votes += 1
                if process_index == 0:
                    print(f"  God decision {god_idx + 1} invalid vote: {voted_spy} - not counted")
        else:
            invalid_votes += 1
            if process_index == 0:
                print(f"  God decision {god_idx + 1} invalid vote structure - not counted")
    
    # Calculate vote statistics
    v_U = vote_counts[spy_player]  # Votes received by spy
    
    # Calculate votes received by each civilian and their average
    civilian_votes = []
    civilian_players = []
    for player_id in range(1, num_players + 1):
        if player_id != spy_player:
            civilian_votes.append(vote_counts[player_id])
            civilian_players.append(player_id)
    
    # Average votes received by civilians
    v_C_bar = sum(civilian_votes) / len(civilian_votes) if civilian_votes else 0
    
    # Calculate suspicion potential
    psi = v_U - v_C_bar  # Ψ = v_U - \bar{v}_C
    delta_psi = psi  # Since previous round potential is 0: ΔΨ = Ψ
    
    # Log vote statistics
    if process_index == 0:
        print(f"[STRATEGIC CLUE] Vote counting results:")
        print(f"  Valid votes from God decisions: {total_god_voters}")
        print(f"  N/A votes (not counted): {na_votes}")
        print(f"  Invalid votes: {invalid_votes}")
        print(f"  Vote counts per player: {dict(vote_counts)}")
        print(f"  v_U (spy votes): {v_U}")
        print(f"  v_C_bar (avg civilian votes): {v_C_bar:.3f}")
        print(f"  Ψ = v_U - v_C_bar = {psi:.3f}")
        print(f"  ΔΨ = {delta_psi:.3f}")
    
    # Calculate rewards with zero-sum guarantee
    rewards = [0.0] * num_players
    
    if process_index == 0:
        print(f"[STRATEGIC CLUE] Reward calculation:")
    for player_id in range(1, num_players + 1):
        player_idx = player_id - 1
        
        if player_id == spy_player:
            # Spy reward: Δ_U = -β * ΔΨ
            rewards[player_idx] = -beta * delta_psi
            if process_index == 0:
                print(f"  Player {player_id} (SPY): {rewards[player_idx]:.3f} = -β*ΔΨ = -{beta}*{delta_psi:.3f}")
        else:
            # Civilian reward: r_{C_j} = Δ_{C_j}^{share} + δ_{C_j}
            
            # Shared camp reward: Δ_{C_j}^{share} = +β * ΔΨ / (num_civilians)
            num_civilians = num_players - 1
            shared_reward = beta * delta_psi / num_civilians
            
            # Individual suspicion reward: δ_{C_j} = -λ * (v_{C_j} - \bar{v}_C)
            v_C_j = vote_counts[player_id]
            individual_suspicion = -lambda_param * (v_C_j - v_C_bar)
            
            rewards[player_idx] = shared_reward + individual_suspicion
            if process_index == 0:
                print(f"  Player {player_id} (civilian): {rewards[player_idx]:.3f} = shared({shared_reward:.3f}) + individual({individual_suspicion:.3f}) [received {v_C_j} votes vs avg {v_C_bar:.3f}]")
    
    # Verify zero-sum property (for debugging)
    total_reward = sum(rewards)
    if process_index == 0:
        print(f"[STRATEGIC CLUE] Total reward: {total_reward:.6f} (should be ~0.0 for zero-sum)")
        if abs(total_reward) > 1e-6:  # Allow small floating point errors
            print(f"[WARNING] Zero-sum violation in clue rewards: total = {total_reward:.6f}")
    
    # Calculate original (raw) metrics for wandb logging
    spy_raw_reward = rewards[spy_player - 1]  # Convert to 0-indexed
    civilian_raw_rewards = [rewards[i] for i in range(len(rewards)) if i != spy_player - 1]
    civilian_raw_mean = sum(civilian_raw_rewards) / len(civilian_raw_rewards) if civilian_raw_rewards else 0.0
    
    # Initialize metrics dictionary
    metrics = {
        'spy_raw_reward': spy_raw_reward,
        'civilian_raw_reward_mean': civilian_raw_mean,
        'suspicion_potential_psi': psi,
        'spy_votes_received': v_U,
        'civilian_votes_avg': v_C_bar,
        'total_valid_votes': total_god_voters,
        'na_votes': na_votes,
        'invalid_votes': invalid_votes
    }
    
    # Apply role advantage adjustment if enabled and generator is provided
    if apply_role_advantage and generator is not None:
        if process_index == 0:
            print(f"[STRATEGIC CLUE] Applying unified role advantage adjustment...")
        
        # Try to get accelerator from the calling context
        accelerator = None
        try:
            import inspect
            frame = inspect.currentframe()
            while frame:
                if 'self' in frame.f_locals and hasattr(frame.f_locals['self'], 'accelerator'):
                    accelerator = frame.f_locals['self'].accelerator
                    break
                frame = frame.f_back
        except:
            pass  # No accelerator found, will use single-GPU mode
        
        # Apply unified role advantage adjustment (uses accelerator for multi-GPU sync)
        adjusted_rewards = generator.apply_unified_role_advantage_adjustment(rewards, spy_player, accelerator, process_index)
        
        # Calculate adjusted metrics for wandb logging
        spy_adjusted_reward = adjusted_rewards[spy_player - 1]  # Convert to 0-indexed
        civilian_adjusted_rewards = [adjusted_rewards[i] for i in range(len(adjusted_rewards)) if i != spy_player - 1]
        civilian_adjusted_mean = sum(civilian_adjusted_rewards) / len(civilian_adjusted_rewards) if civilian_adjusted_rewards else 0.0
        
        # Add adjusted metrics
        metrics.update({
            'spy_adjusted_reward': spy_adjusted_reward,
            'civilian_adjusted_reward_mean': civilian_adjusted_mean,
            'spy_baseline': generator.b_spy,
            'civilian_baseline': generator.b_civ,
            'spy_advantage_adjustment': spy_adjusted_reward - spy_raw_reward,
            'civilian_advantage_adjustment': civilian_adjusted_mean - civilian_raw_mean
        })
        
        if process_index == 0:
            print(f"[STRATEGIC CLUE] Unified role advantage adjustment complete")
            print(f"[STRATEGIC CLUE] Raw rewards: spy={spy_raw_reward:.3f}, civ_mean={civilian_raw_mean:.3f}")
            print(f"[STRATEGIC CLUE] Adjusted rewards: spy={spy_adjusted_reward:.3f}, civ_mean={civilian_adjusted_mean:.3f}")
            print(f"[STRATEGIC CLUE] Baselines: spy={generator.b_spy:.3f}, civ={generator.b_civ:.3f}")
        
        return {
            'rewards': adjusted_rewards,
            'metrics': metrics
        }
    else:
        if process_index == 0 and not apply_role_advantage:
            print(f"[STRATEGIC CLUE] Role advantage adjustment disabled")
        elif process_index == 0 and generator is None:
            print(f"[STRATEGIC CLUE] No generator provided, skipping role advantage adjustment")
        
        # When no role advantage adjustment, adjusted rewards = raw rewards
        metrics.update({
            'spy_adjusted_reward': spy_raw_reward,
            'civilian_adjusted_reward_mean': civilian_raw_mean,
            'spy_baseline': 0.0,
            'civilian_baseline': 0.0,
            'spy_advantage_adjustment': 0.0,
            'civilian_advantage_adjustment': 0.0
        })
        
        return {
            'rewards': rewards,
            'metrics': metrics
        }


def _calculate_enhanced_two_phase_rewards(generator, game_data, clue_completions, decision_completions):
    """
    Enhanced reward calculation using strategic game theory for clue phase and PBRS for decision phase
    
    Args:
        generator: CLEVRSpotDiffGenerator instance
        game_data: Game data including spy information
        clue_completions: List of clue completions for all players across all rounds
        decision_completions: List of decision completions for all players
        
    Returns:
        Tuple of (clue_rewards, decision_rewards, clue_metrics)
    """
    num_players = game_data["num_players"]
    spy_player = game_data["spy_player"]
    
    # Initialize PBRS system for decision phase only
    pbrs = CLEVRPotentialBasedRewardShaping(
        gamma=0.99,        # Discount factor for potential shaping
        lambda_fmt=0.3,    # Format reward weight (increased from 0.2)
        kappa_v=0.0,       # Vote potential coefficient (not used in new system)
        beta_acc=1.2       # Accuracy reward weight
    )
    
    # Extract all votes from decision phase
    all_votes = []
    for completion in decision_completions:
        vote_info = generator.extract_vote_from_decision(completion)
        all_votes.append(vote_info)
    
    # Handle case when only training clue phase (no decision completions available)
    if not decision_completions or len(decision_completions) == 0:
        # Generate simulated votes for strategic clue reward calculation
        # This ensures the strategic game theory mechanism works even during clue-only training
        import random
        
        # Set seed based on game data for reproducible simulation
        game_seed = hash(game_data["game_id"]) % 1000000
        random.seed(game_seed)
        
        spy_player = game_data["spy_player"]
        
        # Simulate realistic voting patterns:
        # - Non-spy players have 60% chance to vote for spy, 40% random
        # - Spy player never votes for themselves (would be strategic suicide)
        simulated_votes = []
        for player_id in range(1, num_players + 1):
            if player_id == spy_player:
                # Spy votes for a random non-spy player
                possible_targets = [p for p in range(1, num_players + 1) if p != spy_player]
                voted_spy = random.choice(possible_targets)
            else:
                # Non-spy player: 60% chance to vote correctly, 40% random
                if random.random() < 0.6:
                    voted_spy = spy_player  # Correct vote
                else:
                    # Random vote (could be wrong)
                    possible_targets = [p for p in range(1, num_players + 1) if p != player_id]
                    voted_spy = random.choice(possible_targets)
            
            simulated_votes.append({
                "voted_spy": voted_spy,
                "reasoning": "Simulated vote for clue-only training"
            })
        
        all_votes = simulated_votes
        
        # Only print from the calling context if we can access accelerator
        # Try to get process index through stack frame inspection
        try:
            import inspect
            frame = inspect.currentframe()
            while frame:
                if 'self' in frame.f_locals and hasattr(frame.f_locals['self'], 'accelerator'):
                    process_index = frame.f_locals['self'].accelerator.process_index
                    break
                frame = frame.f_back
            else:
                process_index = 0  # Default to 0 if not found
        except:
            process_index = 0  # Fallback
        
        if game_seed % 100 == 0 and process_index == 0:  # Debug print for every 100th game
            print(f"[CLUE-ONLY TRAINING] Game {game_data['game_id']}: Simulated votes = {[v['voted_spy'] for v in all_votes]}, Spy = {spy_player}")
    else:
        # Real decision votes available
        # Try to get process index through stack frame inspection
        try:
            import inspect
            frame = inspect.currentframe()
            while frame:
                if 'self' in frame.f_locals and hasattr(frame.f_locals['self'], 'accelerator'):
                    process_index = frame.f_locals['self'].accelerator.process_index
                    break
                frame = frame.f_back
            else:
                process_index = 0  # Default to 0 if not found
        except:
            process_index = 0  # Fallback
        
        if process_index == 0:
            print(f"[CLUE TRAINING] Game {game_data['game_id']}: Real decision votes = {[v.get('voted_spy', 'None') if v else 'None' for v in all_votes]}, Spy = {spy_player}")
    
    # Calculate NEW strategic clue phase rewards based on voting game theory
    clue_result = _calculate_strategic_clue_rewards(
        game_data=game_data,
        all_votes=all_votes,
        num_players=num_players,
        beta=0.1,          # Camp shared potential reward coefficient
        lambda_param=0.1,  # Individual suspicion penalty coefficient
        generator=generator,  # Pass generator instance for role advantage adjustment
        apply_role_advantage=True  # Enable role advantage adjustment
    )
    
    # Extract rewards and metrics from the result
    player_rewards = clue_result['rewards']
    clue_metrics = clue_result['metrics']
    
    # Expand player rewards to all clue rounds
    # For multi-round games, each player gets the same reward for all their clue rounds
    num_rounds = game_data.get("num_rounds", 2)
    clue_rewards = []
    
    # Create rewards for all clue samples: [player1_round1, player2_round1, ..., playerN_round1, player1_round2, ...]
    for round_num in range(num_rounds):
        for player_id in range(1, num_players + 1):
            player_idx = player_id - 1
            clue_rewards.append(player_rewards[player_idx])
    
    # Try to get process index through stack frame inspection for logging
    try:
        import inspect
        frame = inspect.currentframe()
        while frame:
            if 'self' in frame.f_locals and hasattr(frame.f_locals['self'], 'accelerator'):
                process_index = frame.f_locals['self'].accelerator.process_index
                break
            frame = frame.f_back
        else:
            process_index = 0  # Default to 0 if not found
    except:
        process_index = 0  # Fallback
    
    if process_index == 0:
        print(f"[STRATEGIC CLUE] Expanded {len(player_rewards)} player rewards to {len(clue_rewards)} clue samples ({num_rounds} rounds)")
        print(f"[STRATEGIC CLUE] Player rewards: {[f'{r:.3f}' for r in player_rewards]}")
        print(f"[STRATEGIC CLUE] All clue rewards: {[f'{r:.3f}' for r in clue_rewards]}")
    
    # Calculate decision phase rewards using PBRS logic
    decision_rewards = []
    for i, completion in enumerate(decision_completions):
        player_id = i + 1
        
        # Check format quality with flexible answer extraction
        import re
        think_match = re.search(r'<think>(.*?)</think>', completion, re.DOTALL)
        
        # Flexible answer extraction for decision phase
        answer_match = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL)
        if answer_match:
            answer_content = answer_match.group(1).strip()
        else:
            # Try to find <answer> and extract everything after it
            answer_start_match = re.search(r'<answer>\s*(.*)', completion, re.DOTALL)
            if answer_start_match:
                answer_content = answer_start_match.group(1).strip()
                # Clean up any potential closing tags or extra content
                answer_content = re.split(r'</answer>|<think>|<answer>', answer_content)[0].strip()
            else:
                answer_content = ""
        
        if think_match and answer_content:
            think_content = think_match.group(1).strip()
            fmt_ok = len(think_content) > 10 and len(answer_content) > 0
        else:
            fmt_ok = False
        
        # Check accuracy with spy-specific logic
        vote_info = all_votes[i]
        if vote_info and "voted_spy" in vote_info:
            voted_spy = vote_info["voted_spy"]
            
            # Different accuracy reward logic for spy vs non-spy players
            if player_id == spy_player:
                # Spy player: +0.5 if didn't vote for themselves, -1.0 if voted for themselves
                if voted_spy == spy_player:
                    accuracy_component = -1.0  # Spy voted for themselves (self-reveal)
                elif voted_spy == "N/A":
                    accuracy_component = 0.2   # Spy voted N/A (neutral strategy)
                else:
                    accuracy_component = 0.5   # Spy voted for someone else (good strategy)
            else:
                # Non-spy player: +1 for correct guess, -0.5 for N/A, -1 for incorrect
                if voted_spy == spy_player:
                    accuracy_component = 1.0   # Correct identification
                elif voted_spy == "N/A":
                    accuracy_component = -0.5  # Uncertain (better than wrong guess)
                else:
                    accuracy_component = -1.0  # Incorrect identification
        else:
            accuracy_component = -1.0  # No valid vote counts as incorrect
        
        # Calculate PBRS decision reward with custom accuracy component
        # Format reward
        r_fmt = pbrs.lambda_fmt * (2 * int(fmt_ok) - 1)
        
        # Use custom accuracy component instead of standard PBRS accuracy logic
        r_acc = accuracy_component * pbrs.beta_acc
        
        # Potential-based shaping: γ * Φ_next - Φ_prev  
        # Φ_prev = 0 (no previous accuracy state), Φ_next = Φ_acc(correct)
        phi_prev = 0.0
        phi_next = pbrs.phi_acc(accuracy_component > 0)  # Consider positive accuracy as "correct"
        
        shaped_term = pbrs.gamma * phi_next - phi_prev
        
        reward = r_fmt + r_acc + shaped_term
        
        decision_rewards.append(reward)
    
    return clue_rewards, decision_rewards, clue_metrics


def apply_clevr_group_normalization(clue_rewards: list[float], decision_rewards: list[float], 
                                  eps: float = 1e-8) -> tuple[list[float], list[float]]:
    """
    Apply group normalization to CLEVR game rewards within each phase
    
    Args:
        clue_rewards: List of rewards for all players in clue phase
        decision_rewards: List of rewards for all players in decision phase  
        eps: Small constant to prevent division by zero
        
    Returns:
        Tuple of (normalized_clue_rewards, normalized_decision_rewards)
    """
    def normalize_group(rewards: list[float]) -> list[float]:
        if len(rewards) <= 1:
            return rewards
            
        import torch
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        
        group_mean = rewards_tensor.mean()
        group_std = rewards_tensor.std()
        
        if group_std > eps:
            normalized = (rewards_tensor - group_mean) / (group_std + eps)
        else:
            # If std is too small, just center the rewards
            normalized = rewards_tensor - group_mean
            
        return normalized.tolist()
    
    # Apply normalization to each phase separately
    normalized_clue = normalize_group(clue_rewards)
    normalized_decision = normalize_group(decision_rewards)
    
    return normalized_clue, normalized_decision


def _calculate_enhanced_two_phase_rewards_with_group_norm(generator, game_data, clue_completions, decision_completions,
                                                        apply_group_norm: bool = True):
    """
    Enhanced reward calculation with optional group normalization
    
    Args:
        generator: CLEVRSpotDiffGenerator instance
        game_data: Game data including spy information
        clue_completions: List of clue completions for all players across all rounds
        decision_completions: List of decision completions for all players
        apply_group_norm: Whether to apply group normalization to rewards
        
    Returns:
        Tuple of (clue_rewards, decision_rewards, clue_metrics) with optional normalization applied
    """
    # Calculate original rewards using existing function
    clue_rewards, decision_rewards, clue_metrics = _calculate_enhanced_two_phase_rewards(
        generator, game_data, clue_completions, decision_completions
    )
    
    # Apply group normalization if requested
    if apply_group_norm:
        clue_rewards, decision_rewards = apply_clevr_group_normalization(
            clue_rewards, decision_rewards
        )
    
    return clue_rewards, decision_rewards, clue_metrics


def _calculate_single_format_reward(completion: str) -> float:
    """Calculate format reward for a single completion with flexible answer extraction"""
    import re
    
    think_match = re.search(r'<think>(.*?)</think>', completion, re.DOTALL)
    
    # Flexible answer extraction
    answer_match = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL)
    if answer_match:
        answer_content = answer_match.group(1).strip()
    else:
        # Try to find <answer> and extract everything after it
        answer_start_match = re.search(r'<answer>\s*(.*)', completion, re.DOTALL)
        if answer_start_match:
            answer_content = answer_start_match.group(1).strip()
            # Clean up any potential closing tags or extra content
            answer_content = re.split(r'</answer>|<think>|<answer>', answer_content)[0].strip()
        else:
            answer_content = ""
    
    if think_match and answer_content:
        think_content = think_match.group(1).strip()
        
        if len(think_content) > 10 and len(answer_content) > 0:
            return 1.0  # Perfect format with content
        else:
            return 0.5  # Format exists but content is minimal
    elif think_match or answer_content:
        return 0.3  # Partial format
    else:
        return 0.0  # No format


def _calculate_clevr_component_rewards(generator, completions, kwargs, component_type):
    """Helper function to calculate specific component rewards for CLEVR game (backward compatibility)"""
    try:
        # For CLEVR group training, the rewards are already calculated and passed in kwargs
        if 'game_rewards' in kwargs:
            game_rewards = kwargs['game_rewards']
            
            if component_type == "spy_identification":
                # For spy identification, we extract the spy identification component
                # The rewards are already calculated in the trainer
                return game_rewards
            elif component_type == "vote_penalty":
                # For vote penalty, return zero as it's already included in game_rewards
                return [0.0] * len(game_rewards)
            else:  # overall
                # Return total reward
                return game_rewards
        
        # Fallback: if game_rewards not available, try to calculate from completions directly
        print(f"Warning: game_rewards not found for {component_type}, attempting fallback calculation")
        
        # Try to extract game data and calculate rewards
        if 'inputs' not in kwargs:
            print(f"Warning: Missing inputs in kwargs for {component_type}")
            return [0.0] * len(completions)
        
        inputs = kwargs['inputs']
        
        # Group by games and calculate rewards
        games = {}
        completion_idx = 0
        
        for input_item in inputs:
            game_id = input_item["game_data"]["game_id"]
            if game_id not in games:
                games[game_id] = {
                    "game_data": input_item["game_data"],
                    "completions": [],
                    "indices": []
                }
            games[game_id]["completions"].append(completions[completion_idx])
            games[game_id]["indices"].append(completion_idx)
            completion_idx += 1
        
        # Calculate rewards for each game
        all_rewards = [0.0] * len(completions)
        
        for game_id, game_info in games.items():
            game_data = game_info["game_data"]
            game_completions = game_info["completions"]
            indices = game_info["indices"]
            
                    # Extract votes and calculate rewards
        player_votes = []
        for completion in game_completions:
            vote_info = generator.extract_vote_from_decision(completion)
            player_votes.append(vote_info)
            
            # Calculate game rewards
            game_rewards = generator.calculate_game_rewards(game_data, player_votes)
            
            # Assign rewards back to the correct positions
            for reward, idx in zip(game_rewards, indices):
                if component_type == "spy_identification":
                    # Extract spy identification component (first part of reward)
                    all_rewards[idx] = reward if reward >= 0 else 0.0  # Only positive rewards for correct identification
                elif component_type == "vote_penalty":
                    # Extract vote penalty component (negative part of reward)
                    all_rewards[idx] = min(0.0, reward - (1.0 if reward >= 1.0 else 0.0))  # Only penalty part
                else:  # overall
                    all_rewards[idx] = reward
        
        return all_rewards
            
    except Exception as e:
        print(f"Error in CLEVR {component_type} reward: {e}")
        import traceback
        traceback.print_exc()
        return [0.0] * len(completions)


def _calculate_enhanced_two_phase_rewards_with_god_decision(generator, game_data, clue_completions, god_decision_completions):
    """
    Enhanced reward calculation using strategic game theory for clue phase and God's perspective for decision phase
    
    Args:
        generator: CLEVRSpotDiffGenerator instance
        game_data: Game data including spy information
        clue_completions: List of clue completions for all players across all rounds
        god_decision_completions: List of God's perspective decision completions (num_generations copies)
        
    Returns:
        Tuple of (clue_rewards, god_decision_rewards, clue_metrics)
    """
    num_players = game_data["num_players"]
    spy_player = game_data["spy_player"]
    
    # Initialize PBRS system for decision phase only
    pbrs = CLEVRPotentialBasedRewardShaping(
        gamma=0.99,        # Discount factor for potential shaping
        lambda_fmt=0.3,    # Format reward weight (increased from 0.2)
        kappa_v=0.0,       # Vote potential coefficient (not used in new system)
        beta_acc=1.2       # Accuracy reward weight
    )
    
    # Extract votes from God's perspective decision completions
    # Since all God decisions have the same input, we evaluate all num_generations independently
    god_votes = []
    for i, completion in enumerate(god_decision_completions):
        # Debug: Print the actual completion content to see what's being generated
        try:
            import inspect
            frame = inspect.currentframe()
            while frame:
                if 'self' in frame.f_locals and hasattr(frame.f_locals['self'], 'accelerator'):
                    process_index = frame.f_locals['self'].accelerator.process_index
                    break
                frame = frame.f_back
            else:
                process_index = 0  # Default to 0 if not found
        except:
            process_index = 0  # Fallback
        
        if process_index == 0 and i < 3:  # Only print first 3 for debugging
            print(f"[DEBUG] God decision completion {i+1}:")
            print(f"[DEBUG] Raw text length: {len(completion)} chars")
            print(f"[DEBUG] FULL COMPLETION TEXT:")
            print(f"[DEBUG] {'='*60}")
            print(completion)
            print(f"[DEBUG] {'='*60}")
            
            # Check for answer tags specifically
            if '<answer>' in completion:
                answer_start = completion.find('<answer>')
                answer_end = completion.find('</answer>')
                if answer_end != -1:
                    print(f"[DEBUG] Found complete answer: {completion[answer_start:answer_end+9]}")
                else:
                    print(f"[DEBUG] Found answer start but no end: {completion[answer_start:]}")
            else:
                print(f"[DEBUG] No <answer> tag found in completion")
            
            # Check for boxed format
            import re
            boxed_match = re.search(r'\\\\?boxed\{(.*?)\}', completion, re.DOTALL)
            if boxed_match:
                print(f"[DEBUG] Found boxed answer: '{boxed_match.group(1).strip()}'")
            else:
                print(f"[DEBUG] No \\boxed{{}} format found in completion")
            
        vote_info = generator.extract_vote_from_decision(completion, debug_index=i)
        god_votes.append(vote_info)
        
        if process_index == 0 and i < 3:  # Debug the extraction result
            print(f"[DEBUG] Extracted vote_info: {vote_info}")
            print(f"[DEBUG] ---")

    # Handle case when only training clue phase (no god decision completions available)
    if not god_decision_completions or len(god_decision_completions) == 0:
        # Generate simulated votes for strategic clue reward calculation
        # This ensures the strategic game theory mechanism works even during clue-only training
        import random
        
        # Set seed based on game data for reproducible simulation
        game_seed = hash(game_data["game_id"]) % 1000000
        random.seed(game_seed)
        
        spy_player = game_data["spy_player"]
        
        # Simulate realistic voting patterns:
        # - Non-spy players have 60% chance to vote for spy, 40% random
        # - Spy player never votes for themselves (would be strategic suicide)
        simulated_votes = []
        for player_id in range(1, num_players + 1):
            if player_id == spy_player:
                # Spy votes for a random non-spy player
                possible_targets = [p for p in range(1, num_players + 1) if p != spy_player]
                voted_spy = random.choice(possible_targets)
            else:
                # Non-spy player: 60% chance to vote correctly, 40% random
                if random.random() < 0.6:
                    voted_spy = spy_player  # Correct vote
                else:
                    # Random vote (could be wrong)
                    possible_targets = [p for p in range(1, num_players + 1) if p != player_id]
                    voted_spy = random.choice(possible_targets)
            
            simulated_votes.append({
                "voted_spy": voted_spy,
                "reasoning": "Simulated vote for clue-only training"
            })
        
        god_votes = simulated_votes
        
        # Only print from the calling context if we can access accelerator
        # Try to get process index through stack frame inspection
        try:
            import inspect
            frame = inspect.currentframe()
            while frame:
                if 'self' in frame.f_locals and hasattr(frame.f_locals['self'], 'accelerator'):
                    process_index = frame.f_locals['self'].accelerator.process_index
                    break
                frame = frame.f_back
            else:
                process_index = 0  # Default to 0 if not found
        except:
            process_index = 0  # Fallback
        
        if game_seed % 100 == 0 and process_index == 0:  # Debug print for every 100th game
            print(f"[CLUE-ONLY TRAINING] Game {game_data['game_id']}: Simulated votes = {[v['voted_spy'] for v in god_votes]}, Spy = {spy_player}")
    else:
        # Real God decision votes available
        # Try to get process index through stack frame inspection
        try:
            import inspect
            frame = inspect.currentframe()
            while frame:
                if 'self' in frame.f_locals and hasattr(frame.f_locals['self'], 'accelerator'):
                    process_index = frame.f_locals['self'].accelerator.process_index
                    break
                frame = frame.f_back
            else:
                process_index = 0  # Default to 0 if not found
        except:
            process_index = 0  # Fallback
        
        if process_index == 0:
            print(f"[GOD CLUE TRAINING] Game {game_data['game_id']}: God decision votes = {[v.get('voted_spy', 'None') if v else 'None' for v in god_votes]}, Spy = {spy_player}")
    
    # Calculate NEW strategic clue phase rewards based on voting game theory
    # Use God votes instead of player votes for strategic calculation
    clue_result = _calculate_strategic_clue_rewards(
        game_data=game_data,
        all_votes=god_votes,  # Use God votes instead of player votes
        num_players=num_players,
        beta=0.1,          # Camp shared potential reward coefficient
        lambda_param=0.1,  # Individual suspicion penalty coefficient
        generator=generator,  # Pass generator instance for role advantage adjustment
        apply_role_advantage=True  # Enable role advantage adjustment
    )
    
    # Extract rewards and metrics from the result
    player_rewards = clue_result['rewards']
    clue_metrics = clue_result['metrics']
    
    # Expand player rewards to all clue rounds
    # For multi-round games, each player gets the same reward for all their clue rounds
    num_rounds = game_data.get("num_rounds", 2)
    clue_rewards = []
    
    # Create rewards for all clue samples: [player1_round1, player2_round1, ..., playerN_round1, player1_round2, ...]
    for round_num in range(num_rounds):
        for player_id in range(1, num_players + 1):
            player_idx = player_id - 1
            clue_rewards.append(player_rewards[player_idx])
    
    # Try to get process index through stack frame inspection for logging
    try:
        import inspect
        frame = inspect.currentframe()
        while frame:
            if 'self' in frame.f_locals and hasattr(frame.f_locals['self'], 'accelerator'):
                process_index = frame.f_locals['self'].accelerator.process_index
                break
            frame = frame.f_back
        else:
            process_index = 0  # Default to 0 if not found
    except:
        process_index = 0  # Fallback
    
    if process_index == 0:
        print(f"[GOD STRATEGIC CLUE] Expanded {len(player_rewards)} player rewards to {len(clue_rewards)} clue samples ({num_rounds} rounds)")
        print(f"[GOD STRATEGIC CLUE] Player rewards: {[f'{r:.3f}' for r in player_rewards]}")
        print(f"[GOD STRATEGIC CLUE] All clue rewards: {[f'{r:.3f}' for r in clue_rewards]}")
    
    # Calculate God decision phase rewards using PBRS logic
    # Each God decision completion gets evaluated independently
    god_decision_rewards = []
    for i, completion in enumerate(god_decision_completions):
        
        # Check format quality with flexible answer extraction for new format
        import re
        
        # Extract thinking and answer content using flexible format support
        thinking_content = generator.extract_thinking_from_clue(completion)
        boxed_match = re.search(r'\\\\?boxed\{(.*?)\}', completion, re.DOTALL)
        
        if boxed_match:
            answer_content = boxed_match.group(1).strip()
        else:
            # Fallback to old format for backward compatibility
            answer_match = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL)
            if answer_match:
                answer_content = answer_match.group(1).strip()
            else:
                # Try to find <answer> and extract everything after it
                answer_start_match = re.search(r'<answer>\s*(.*)', completion, re.DOTALL)
                if answer_start_match:
                    answer_content = answer_start_match.group(1).strip()
                    # Clean up any potential closing tags or extra content
                    answer_content = re.split(r'</answer>|<think>|<answer>', answer_content)[0].strip()
                else:
                    answer_content = ""
        
        # Check format: need meaningful thinking content and valid answer
        if thinking_content and answer_content and len(thinking_content) > 10:
            fmt_ok = True
        else:
            fmt_ok = False
        
        # Check accuracy for God's perspective (God should identify the spy correctly)
        vote_info = god_votes[i] if i < len(god_votes) else None
        if vote_info and "voted_spy" in vote_info:
            voted_spy = vote_info["voted_spy"]
            
            # God perspective accuracy: +1 for correct identification, -0.5 for N/A, -1 for incorrect
            if voted_spy == spy_player:
                accuracy_component = 1.0   # Correct identification
            elif voted_spy == "N/A":
                accuracy_component = -0.5  # Uncertain (partial credit)
            else:
                accuracy_component = -1.0  # Incorrect identification
        else:
            accuracy_component = -1.0  # No valid vote
        
        # Calculate PBRS decision reward with God-specific accuracy component
        # Format reward
        r_fmt = pbrs.lambda_fmt * (2 * int(fmt_ok) - 1)
        
        # Use God-specific accuracy component
        r_acc = accuracy_component * pbrs.beta_acc
        
        # Potential-based shaping: γ * Φ_next - Φ_prev  
        # Φ_prev = 0 (no previous accuracy state), Φ_next = Φ_acc(correct)
        phi_prev = 0.0
        phi_next = pbrs.phi_acc(accuracy_component > 0)  # Consider positive accuracy as "correct"
        
        shaped_term = pbrs.gamma * phi_next - phi_prev
        
        reward = r_fmt + r_acc + shaped_term
        
        god_decision_rewards.append(reward)
    
    if process_index == 0:
        print(f"[GOD DECISION] Generated {len(god_decision_rewards)} God decision rewards: {[f'{r:.3f}' for r in god_decision_rewards]}")
    
    return clue_rewards, god_decision_rewards, clue_metrics


if __name__ == "__main__":
    # Test the generator
    generator = CLEVRSpotDiffGenerator()
    
    # Generate a sample
    game_data = generator.generate_game_data(epoch=1, sample_idx=1)
    print("Generated game data:")
    print(f"Game ID: {game_data['game_id']}")
    print(f"Spy player: {game_data['spy_player']}")
    print(f"Num players: {game_data['num_players']}")
    
    # Test clue phase
    clue_sample = generator.format_clue_phase_sample(game_data, 1, 1, "", "")
    print(f"Clue sample for player 1: {clue_sample['conversations'][0]['value'][:100]}...")
    
    # Test decision phase
    decision_sample = generator.format_decision_phase_sample(
        game_data, 1, "Round 1:\nPLAYER 1: The blue sphere is in front of the red cube\nPLAYER 2: The green cylinder is to the left", 
        "I need to identify the odd player"
    )
    print(f"Decision sample for player 1: {decision_sample['conversations'][0]['value'][:100]}...")
    
    # Test new reward mechanism
    print("\n=== Testing New Reward Mechanism ===")
    
    # Test case 1: Perfect scenario
    test_game_data = {
        "num_players": 4,
        "spy_player": 3  # Player 3 is the spy
    }
    
    # Scenario: All non-spy players correctly identify spy, spy gets no votes for themselves
    test_votes = [
        {"voted_spy": 3, "reasoning": "Direct number vote"},    # Player 1 (detective)
        {"voted_spy": 3, "reasoning": "Direct number vote"},    # Player 2 (detective) 
        {"voted_spy": 1, "reasoning": "Direct number vote"}, # Player 3 (spy)
        {"voted_spy": 3, "reasoning": "Direct number vote"}     # Player 4 (detective)
    ]
    
    rewards = generator.calculate_game_rewards(test_game_data, test_votes)
    print(f"\nTest Case 1 - Spy is caught:")
    print(f"Game data: {test_game_data}")
    print(f"Votes: {[v['voted_spy'] for v in test_votes]}")
    print(f"Rewards: {rewards}")
    print(f"Player 1 (detective): {rewards[0]:.3f}")
    print(f"Player 2 (detective): {rewards[1]:.3f}") 
    print(f"Player 3 (spy): {rewards[2]:.3f}")
    print(f"Player 4 (detective): {rewards[3]:.3f}")
    
    # Test case 2: Spy successfully misleads
    test_votes_2 = [
        {"voted_spy": 2, "reasoning": "Direct number vote"},    # Player 1 (detective) - wrong
        {"voted_spy": 1, "reasoning": "Direct number vote"},    # Player 2 (detective) - wrong
        {"voted_spy": 1, "reasoning": "Direct number vote"},   # Player 3 (spy) - misleading
        {"voted_spy": 1, "reasoning": "Direct number vote"}     # Player 4 (detective) - wrong
    ]
    
    rewards_2 = generator.calculate_game_rewards(test_game_data, test_votes_2)
    print(f"\nTest Case 2 - Spy successfully misleads:")
    print(f"Votes: {[v['voted_spy'] for v in test_votes_2]}")
    print(f"Rewards: {rewards_2}")
    print(f"Player 1 (detective): {rewards_2[0]:.3f}")
    print(f"Player 2 (detective): {rewards_2[1]:.3f}") 
    print(f"Player 3 (spy): {rewards_2[2]:.3f}")
    print(f"Player 4 (detective): {rewards_2[3]:.3f}")
    
    # Test case 3: Spy votes for themselves (self-reveal)
    test_votes_3 = [
        {"voted_spy": 3, "reasoning": "Direct number vote"},    # Player 1 (detective)
        {"voted_spy": 3, "reasoning": "Direct number vote"},    # Player 2 (detective) 
        {"voted_spy": 3, "reasoning": "Direct number vote"},     # Player 3 (spy) - self-reveal penalty
        {"voted_spy": 3, "reasoning": "Direct number vote"}     # Player 4 (detective)
    ]
    
    rewards_3 = generator.calculate_game_rewards(test_game_data, test_votes_3)
    print(f"\nTest Case 3 - Spy self-reveals:")
    print(f"Votes: {[v['voted_spy'] for v in test_votes_3]}")
    print(f"Rewards: {rewards_3}")
    print(f"Player 1 (detective): {rewards_3[0]:.3f}")
    print(f"Player 2 (detective): {rewards_3[1]:.3f}") 
    print(f"Player 3 (spy): {rewards_3[2]:.3f}")
    print(f"Player 4 (detective): {rewards_3[3]:.3f}")
    
    # Test vote extraction with new number format
    print("\n=== Testing Vote Extraction ===")
    test_responses = [
        "<think>I think player 2 is suspicious</think><answer>2</answer>",
        "<think>Based on clues, player 1 seems odd</think><answer>1</answer>", 
        "<answer>3</answer>",
        "I vote for player 2",  # No answer tags
        "<think>Player 3 is the spy</think><answer>Player 3</answer>",  # With "Player" text
    ]
    
    for i, response in enumerate(test_responses):
        vote = generator.extract_vote_from_decision(response)
        print(f"Response {i+1}: {response[:50]}...")
        print(f"Extracted vote: {vote}")
    
    print("\n=== Reward Mechanism Analysis ===")
    print("Detective rewards = Case-solving score + Suspicion penalty")
    print("  - Case-solving: +1 correct, -1 incorrect")
    print("  - Suspicion: -0.6 * votes_received / (N-1)")
    print("\nSpy rewards = Stealth score + Misleading bonus + Self-reveal penalty")
    print("  - Stealth: +1 - votes_received / (N-1)")
    print("  - Misleading: +0.5 * wrong_votes / (N-1)")
    print("  - Self-reveal: -1 if votes for themselves")
    print("  - All rewards clipped to [-1, 1]")
    print("\n=== NEW Decision Phase Accuracy Rewards ===")
    print("Non-spy players:")
    print("  - Vote correctly (identify spy): +1.0")
    print("  - Vote incorrectly: -1.0")
    print("Spy players:")
    print("  - Vote for someone else (avoid self-reveal): +0.5")
    print("  - Vote for themselves (self-reveal): -1.0")
    
    print("\n=== Testing Enhanced Clue Reward System ===")
    
    # Test different types of clues
    test_clues = [
        # High quality clue - contains color, shape, position
        "<think>I need to describe my image precisely. I see a red cube to the left of a blue sphere.</think><answer>The red cube is to the left of the blue sphere.</answer>",
        
        # Medium quality clue - contains shape and position but no color
        "<think>I should mention the spatial relationship.</think><answer>The cube is next to the cylinder.</answer>",
        
        # Low quality clue - vague description
        "<think>I see some objects.</think><answer>There are some things in the picture.</answer>",
        
        # Poor format clue - missing think tags
        "<answer>The ball is big.</answer>",
        
        # High quality with size and comparative language
        "<think>I need to compare sizes and mention positions.</think><answer>The large red sphere is bigger than the small blue cube beside it.</answer>",
        
        # Vague but formatted clue
        "<think>I see various objects in the scene.</think><answer>I can see several objects in the image.</answer>",
        
        # GAME LEAK - mentions "odd" (SEVERE PENALTY)
        "<think>This looks odd to me.</think><answer>The yellow cube is odd compared to others.</answer>",
        
        # GAME LEAK - mentions "player" (SEVERE PENALTY)
        "<think>Other players might notice this.</think><answer>The green sphere is what other players would see.</answer>",
        
        # GAME LEAK - mentions "different" (SEVERE PENALTY)
        "<think>My image might be different.</think><answer>The blue cylinder looks different from what I expected.</answer>",
        
        # GAME LEAK - mentions multiple game words (VERY SEVERE PENALTY)
        "<think>I suspect this is the spy image.</think><answer>I think this odd picture is different from other players.</answer>",
        
        # Good quality clue without game leaks
        "<think>I need to describe the spatial arrangement clearly.</think><answer>The purple cube sits behind the orange sphere.</answer>",
        
        # REPETITION TEST - exact repeat (SEVERE PENALTY)
        "<think>I need to describe my image precisely. I see a red cube to the left of a blue sphere.</think><answer>The red cube is to the left of the blue sphere.</answer>",
        
        # REPETITION TEST - very similar (HIGH PENALTY)
        "<think>Let me describe the spatial relationship.</think><answer>The red cube is positioned to the left of the blue sphere.</answer>",
        
        # REPETITION TEST - moderate similarity (MEDIUM PENALTY)
        "<think>I should mention the objects I see.</think><answer>The cube and sphere are arranged with the cube on the left.</answer>",
        
        # ORIGINALITY TEST - different but valid clue (NO PENALTY)
        "<think>I'll focus on different objects this time.</think><answer>The green cylinder stands upright near the corner.</answer>"
    ]
    
    print("\nTesting clue rewards:")
    enhanced_rewards = _calculate_enhanced_clevr_clue_rewards(test_clues)
    
    for i, (clue, reward) in enumerate(zip(test_clues, enhanced_rewards)):
        print(f"\nClue {i+1} (Reward: {reward:.3f}):")
        # Extract answer part for display
        import re
        answer_match = re.search(r'<answer>(.*?)</answer>', clue, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
            print(f"  Answer: {answer}")
        else:
            print(f"  Raw: {clue[:60]}...")
        
        # Analyze reward components
        if reward >= 0.8:
            print("  Quality: EXCELLENT - Rich vocabulary and clear description")
        elif reward >= 0.6:
            print("  Quality: GOOD - Contains specific CLEVR elements")
        elif reward >= 0.4:
            print("  Quality: FAIR - Basic format but lacking specificity")
        elif reward >= 0.2:
            print("  Quality: POOR - Minimal content or vague language")
        elif reward < 0:
            print("  Quality: FORBIDDEN - Contains game information leakage!")
        else:
            print("  Quality: UNACCEPTABLE - No proper format")
        
        # Check for game leaks in the test
        answer_lower = ""
        if answer_match:
            answer_lower = answer_match.group(1).strip().lower()
        
        game_leak_words = {
            'odd', 'player', 'spy', 'different', 'same', 'vote', 'voting', 'suspect', 
            'suspicious', 'game', 'round', 'clue', 'guess', 'guessing', 'identify',
            'who is', 'which one', 'which player', 'other players', 'others',
            'i am', 'you are', 'we are', 'they are', 'everyone', 'somebody',
            'modified', 'original', 'changed', 'replaced', 'switch', 'switched'
        }
        
        detected_leaks = [word for word in game_leak_words if word in answer_lower]
        if detected_leaks:
            print(f"  🚨 GAME LEAKS DETECTED: {detected_leaks} (Penalty: -0.5 each)")
        
        # Check for repetition in the test
        if answer_match and i > 0:  # Only check for clues after the first one
            from difflib import SequenceMatcher
            current_answer_test = answer_match.group(1).strip().lower()
            max_similarity = 0.0
            most_similar_idx = -1
            
            # Check against all previous answers
            for prev_i in range(i):
                prev_answer_match = re.search(r'<answer>(.*?)</answer>', test_clues[prev_i], re.DOTALL)
                if prev_answer_match:
                    prev_answer_test = prev_answer_match.group(1).strip().lower()
                    similarity = SequenceMatcher(None, current_answer_test, prev_answer_test).ratio()
                    if similarity > max_similarity:
                        max_similarity = similarity
                        most_similar_idx = prev_i
            
            if max_similarity >= 0.7:
                print(f"  🔄 SEVERE REPETITION: {max_similarity:.2f} similarity with Clue {most_similar_idx + 1}")
            elif max_similarity >= 0.5:
                print(f"  🔄 HIGH REPETITION: {max_similarity:.2f} similarity with Clue {most_similar_idx + 1}")
            elif max_similarity >= 0.3:
                print(f"  🔄 MODERATE REPETITION: {max_similarity:.2f} similarity with Clue {most_similar_idx + 1}")
    
    print(f"\nReward statistics:")
    print(f"  Mean reward: {sum(enhanced_rewards)/len(enhanced_rewards):.3f}")
    print(f"  Max reward: {max(enhanced_rewards):.3f}")
    print(f"  Min reward: {min(enhanced_rewards):.3f}")
    print(f"  Range: {max(enhanced_rewards) - min(enhanced_rewards):.3f}")
    
    print("\n=== Enhanced Reward Features ===")
    print("NEW Clue Phase Rewards:")
    print("✓ Base format reward (0.2) + content analysis")
    print("✓ Color vocabulary detection (+0.20)")
    print("✓ Shape vocabulary detection (+0.25)")  
    print("✓ Position vocabulary detection (+0.20)")
    print("✓ Size vocabulary detection (+0.15)")
    print("✓ Spatial relations detection (+0.10)")
    print("✓ Multi-category bonus (+0.1 per additional category)")
    print("✓ Comparative language bonus (+0.05 per pattern)")
    print("✓ Specific object identification bonus (+0.05)")
    print("🚨 SEVERE game leak penalty (-0.5 per forbidden word)")
    print("🔄 REPETITION penalties (prevent copying previous clues):")
    print("    - Severe repetition (≥70% similar): -0.8")
    print("    - High repetition (≥50% similar): -0.4") 
    print("    - Moderate repetition (≥30% similar): -0.2")
    print("    - Word overlap (≥60% overlap): -0.5")
    print("✗ Vague language penalty (-0.08 per vague phrase)")
    print("✗ Length penalties (too short/long: -0.1)")
    print("\nFORBIDDEN WORDS (Game Information Leakage):")
    print("  'odd', 'player', 'spy', 'different', 'same', 'vote', 'suspect',")
    print("  'game', 'round', 'clue', 'guess', 'identify', 'who is', 'others',")
    print("  'modified', 'original', 'changed', 'replaced', 'switch', etc.")
    print("\nThis encourages detailed, specific descriptions while preventing:")
    print("  🚨 Game information leakage (mentioning game mechanics)")
    print("  🔄 Repetitive clues (copying previous statements)")
    print("  ❌ Vague or unhelpful descriptions")
    print("Result: Forces models to be creative, specific, and strategic!")
    
    print("\n" + "="*60)
    print("=== Testing NEW Strategic Clue Phase Reward Mechanism ===")
    print("="*60)
    
    # Test the new strategic clue reward system
    test_strategic_game_data = {
        "num_players": 4,
        "spy_player": 2  # Player 2 is the spy
    }
    
    print(f"\nGame Setup: {test_strategic_game_data['num_players']} players, Player {test_strategic_game_data['spy_player']} is spy")
    print("Formula:")
    print("1. Count votes from non-spy players only (3 non-spy players)")
    print("2. Suspicion potential: Ψ = v_U - \\bar{v}_C")
    print("3. Camp rewards: Spy = -β*ΔΨ, Civilians = +β*ΔΨ/3")
    print("4. Individual suspicion: δ_{C_j} = -λ*(v_{C_j} - \\bar{v}_C)")
    print("5. Zero-sum guarantee: Total rewards = 0")
    print("6. Parameters: β = λ = 0.1")
    
    # Test Case 1: Spy successfully deceives (gets few votes)
    print(f"\n--- Test Case 1: Spy Successfully Deceives ---")
    strategic_votes_1 = [
        {"voted_spy": 3, "reasoning": "Player 1 (non-spy) votes for Player 3"},  # Player 1 
        {"voted_spy": 1, "reasoning": "Player 2 (SPY) votes for Player 1"},      # Player 2 (SPY)
        {"voted_spy": 1, "reasoning": "Player 3 (non-spy) votes for Player 1"},  # Player 3 
        {"voted_spy": 4, "reasoning": "Player 4 (non-spy) votes for Player 4"}   # Player 4
    ]
    
    strategic_rewards_1 = _calculate_strategic_clue_rewards(
        test_strategic_game_data, strategic_votes_1, 4, beta=0.1, lambda_param=0.1, 
        generator=None, apply_role_advantage=False  # Disable for testing basic functionality
    )
    
    print("Voting results (only non-spy votes count):")
    print("  Player 1 (non-spy): votes for Player 3")
    print("  Player 2 (SPY): votes for Player 1 (ignored)")  
    print("  Player 3 (non-spy): votes for Player 1")
    print("  Player 4 (non-spy): votes for Player 4")
    print()
    print("Vote counts from non-spy players:")
    # Calculate manually for display
    votes_from_nonspy_1 = {"1": 1, "2": 0, "3": 1, "4": 1}  # Player 2 is spy, ignore their vote
    v_U_1 = votes_from_nonspy_1["2"]  # 0 votes
    v_C_bar_1 = (votes_from_nonspy_1["1"] + votes_from_nonspy_1["3"] + votes_from_nonspy_1["4"]) / 3  # (1+1+1)/3 = 1
    psi_1 = v_U_1 - v_C_bar_1  # 0 - 1 = -1
    
    print(f"  v_U (spy votes): {v_U_1}")
    print(f"  v_C_bar (avg civilian votes): {v_C_bar_1:.3f}")
    print(f"  Ψ = v_U - v_C_bar = {psi_1:.3f}")
    print()
    print("Strategic rewards:")
    for i, reward in enumerate(strategic_rewards_1):
        player_id = i + 1
        if player_id == test_strategic_game_data["spy_player"]:
            print(f"  Player {player_id} (SPY): {reward:.3f} = -β*ΔΨ = -0.1*({psi_1:.3f}) = {-0.1*psi_1:.3f}")
        else:
            v_j = votes_from_nonspy_1[str(player_id)]
            shared = 0.1 * psi_1 / 3
            individual = -0.1 * (v_j - v_C_bar_1)
            print(f"  Player {player_id} (civilian): {reward:.3f} = shared({shared:.3f}) + individual({individual:.3f})")
    
    total_1 = sum(strategic_rewards_1)
    print(f"  Total reward: {total_1:.6f} (should be ~0.0 for zero-sum)")
    
    # Test Case 2: Spy gets caught (receives many votes)
    print(f"\n--- Test Case 2: Spy Gets Caught ---")
    strategic_votes_2 = [
        {"voted_spy": 2, "reasoning": "Player 1 (non-spy) votes for Player 2"},  # Player 1 
        {"voted_spy": 3, "reasoning": "Player 2 (SPY) votes for Player 3"},      # Player 2 (SPY)
        {"voted_spy": 2, "reasoning": "Player 3 (non-spy) votes for Player 2"},  # Player 3 
        {"voted_spy": 2, "reasoning": "Player 4 (non-spy) votes for Player 2"}   # Player 4
    ]
    
    strategic_rewards_2 = _calculate_strategic_clue_rewards(
        test_strategic_game_data, strategic_votes_2, 4, beta=0.1, lambda_param=0.1,
        generator=None, apply_role_advantage=False  # Disable for testing basic functionality
    )
    
    print("Voting results (only non-spy votes count):")
    print("  Player 1 (non-spy): votes for Player 2 (SPY)")
    print("  Player 2 (SPY): votes for Player 3 (ignored)")  
    print("  Player 3 (non-spy): votes for Player 2 (SPY)")
    print("  Player 4 (non-spy): votes for Player 2 (SPY)")
    print()
    print("Vote counts from non-spy players:")
    votes_from_nonspy_2 = {"1": 0, "2": 3, "3": 0, "4": 0}  # All 3 non-spy players vote for spy
    v_U_2 = votes_from_nonspy_2["2"]  # 3 votes
    v_C_bar_2 = (votes_from_nonspy_2["1"] + votes_from_nonspy_2["3"] + votes_from_nonspy_2["4"]) / 3  # (0+0+0)/3 = 0
    psi_2 = v_U_2 - v_C_bar_2  # 3 - 0 = 3
    
    print(f"  v_U (spy votes): {v_U_2}")
    print(f"  v_C_bar (avg civilian votes): {v_C_bar_2:.3f}")
    print(f"  Ψ = v_U - v_C_bar = {psi_2:.3f}")
    print()
    print("Strategic rewards:")
    for i, reward in enumerate(strategic_rewards_2):
        player_id = i + 1
        if player_id == test_strategic_game_data["spy_player"]:
            print(f"  Player {player_id} (SPY): {reward:.3f} = -β*ΔΨ = -0.1*({psi_2:.3f}) = {-0.1*psi_2:.3f}")
        else:
            v_j = votes_from_nonspy_2[str(player_id)]
            shared = 0.1 * psi_2 / 3
            individual = -0.1 * (v_j - v_C_bar_2)
            print(f"  Player {player_id} (civilian): {reward:.3f} = shared({shared:.3f}) + individual({individual:.3f})")
    
    total_2 = sum(strategic_rewards_2)
    print(f"  Total reward: {total_2:.6f} (should be ~0.0 for zero-sum)")
    
    # Test Case 3: Mixed voting with individual differences
    print(f"\n--- Test Case 3: Mixed Voting (Individual Suspicion Differences) ---")
    strategic_votes_3 = [
        {"voted_spy": 3, "reasoning": "Player 1 (non-spy) votes for Player 3"},  # Player 1 
        {"voted_spy": 1, "reasoning": "Player 2 (SPY) votes for Player 1"},      # Player 2 (SPY)
        {"voted_spy": 2, "reasoning": "Player 3 (non-spy) votes for Player 2"},  # Player 3 
        {"voted_spy": 3, "reasoning": "Player 4 (non-spy) votes for Player 3"}   # Player 4
    ]
    
    strategic_rewards_3 = _calculate_strategic_clue_rewards(
        test_strategic_game_data, strategic_votes_3, 4, beta=0.1, lambda_param=0.1,
        generator=None, apply_role_advantage=False  # Disable for testing basic functionality
    )
    
    print("Voting results (only non-spy votes count):")
    print("  Player 1 (non-spy): votes for Player 3")
    print("  Player 2 (SPY): votes for Player 1 (ignored)")  
    print("  Player 3 (non-spy): votes for Player 2 (SPY)")
    print("  Player 4 (non-spy): votes for Player 3")
    print()
    print("Vote counts from non-spy players:")
    votes_from_nonspy_3 = {"1": 0, "2": 1, "3": 2, "4": 0}  
    v_U_3 = votes_from_nonspy_3["2"]  # 1 vote
    v_C_bar_3 = (votes_from_nonspy_3["1"] + votes_from_nonspy_3["3"] + votes_from_nonspy_3["4"]) / 3  # (0+2+0)/3 = 2/3
    psi_3 = v_U_3 - v_C_bar_3  # 1 - 2/3 = 1/3
    
    print(f"  v_U (spy votes): {v_U_3}")
    print(f"  v_C_bar (avg civilian votes): {v_C_bar_3:.3f}")
    print(f"  Ψ = v_U - v_C_bar = {psi_3:.3f}")
    print()
    print("Strategic rewards (note individual suspicion differences):")
    for i, reward in enumerate(strategic_rewards_3):
        player_id = i + 1
        if player_id == test_strategic_game_data["spy_player"]:
            print(f"  Player {player_id} (SPY): {reward:.3f} = -β*ΔΨ = -0.1*({psi_3:.3f}) = {-0.1*psi_3:.3f}")
        else:
            v_j = votes_from_nonspy_3[str(player_id)]
            shared = 0.1 * psi_3 / 3
            individual = -0.1 * (v_j - v_C_bar_3)
            print(f"  Player {player_id} (civilian): {reward:.3f} = shared({shared:.3f}) + individual({individual:.3f}) [received {v_j} votes vs avg {v_C_bar_3:.3f}]")
    
    total_3 = sum(strategic_rewards_3)
    print(f"  Total reward: {total_3:.6f} (should be ~0.0 for zero-sum)")
    
    # Test Case 4: Testing N/A votes (should NOT count toward anyone)
    print(f"\n--- Test Case 4: N/A Votes Should Not Count ---")
    strategic_votes_4 = [
        {"voted_spy": 2, "reasoning": "Player 1 (non-spy) votes for Player 2"},  # Player 1 
        {"voted_spy": 1, "reasoning": "Player 2 (SPY) votes for Player 1"},      # Player 2 (SPY)
        {"voted_spy": 2, "reasoning": "Player 3 (non-spy) votes for Player 2"},  # Player 3 
        {"voted_spy": "N/A", "reasoning": "Player 4 (non-spy) votes N/A"}        # Player 4 (N/A)
    ]
    
    strategic_rewards_4 = _calculate_strategic_clue_rewards(
        test_strategic_game_data, strategic_votes_4, 4, beta=0.1, lambda_param=0.1,
        generator=None, apply_role_advantage=False  # Disable for testing basic functionality
    )
    
    print("Expected behavior:")
    print("  - Only Players 1 and 3 votes should count (both vote for spy)")
    print("  - Player 4's N/A vote should NOT count toward anyone")
    print("  - Spy should receive 2 votes (not 3)")
    print("  - Average civilian votes should be calculated correctly")
    print()
    
    # Manual calculation for verification
    votes_from_nonspy_4 = {"1": 0, "2": 2, "3": 0, "4": 0}  # Only count valid votes, ignore N/A
    v_U_4 = votes_from_nonspy_4["2"]  # 2 votes (not 3!)
    v_C_bar_4 = (votes_from_nonspy_4["1"] + votes_from_nonspy_4["3"] + votes_from_nonspy_4["4"]) / 3  # (0+0+0)/3 = 0
    psi_4 = v_U_4 - v_C_bar_4  # 2 - 0 = 2
    
    print("Expected calculation:")
    print(f"  Valid non-spy votes: Player 1→2, Player 3→2 (Player 4→N/A ignored)")
    print(f"  v_U (spy votes): {v_U_4}")
    print(f"  v_C_bar (avg civilian votes): {v_C_bar_4:.3f}")
    print(f"  Ψ = v_U - v_C_bar = {psi_4:.3f}")
    print()
    print("Strategic rewards (N/A test):")
    for i, reward in enumerate(strategic_rewards_4):
        player_id = i + 1
        if player_id == test_strategic_game_data["spy_player"]:
            expected = -0.1 * psi_4
            print(f"  Player {player_id} (SPY): {reward:.3f} (expected: {expected:.3f})")
        else:
            v_j = votes_from_nonspy_4[str(player_id)]
            shared = 0.1 * psi_4 / 3
            individual = -0.1 * (v_j - v_C_bar_4)
            expected = shared + individual
            print(f"  Player {player_id} (civilian): {reward:.3f} (expected: {expected:.3f})")
    
    total_4 = sum(strategic_rewards_4)
    print(f"  Total reward: {total_4:.6f} (should be ~0.0 for zero-sum)")
    
    print(f"\n" + "="*60)
    print("=== Strategic Clue Reward Analysis ===")
    print("="*60)
    print("🎯 NEW CLUE PHASE = STRATEGIC GAME THEORY (Zero-Sum)")
    print("📊 Reward Components:")
    print("   1. Camp shared reward: Spy vs Civilians competition")
    print("   2. Individual suspicion: Penalty for being suspected more than average")
    print("   3. Zero-sum guarantee: Total rewards always = 0")
    print()
    print("🎮 Strategic Incentives:")
    print("   SPY: Minimize own votes (be convincing), maximize civilian confusion")
    print("   CIVILIANS: Help identify spy while avoiding personal suspicion")
    print()
    print("⚖️  Zero-Sum Properties:")
    print("   - No matter how players vote, total reward = 0")
    print("   - One camp's gain = other camp's loss") 
    print("   - Individual differences within camps based on suspicion")
    print()
    print("🔧 Parameters:")
    print("   β = 0.1: Camp shared reward coefficient")
    print("   λ = 0.1: Individual suspicion penalty coefficient")
    print()
    print("✅ OLD vs NEW:")
    print("   OLD: Content quality + vocabulary + format rewards")
    print("   NEW: Pure strategic game theory based on voting outcomes")
    print("   Decision phase: Unchanged (still uses accuracy + format)")
    print()
    print("🚀 Expected Behavior:")
    print("   - Spy learns to blend in and mislead")
    print("   - Civilians learn to give distinctive clues without standing out")
    print("   - Strategic communication emerges naturally")
    print("   - Zero-sum creates true competitive dynamics") 
    
    print("\n" + "="*60)
    print("=== Testing NEW Role Advantage Baseline System ===")
    print("="*60)
    
    # Test the new role advantage baseline system
    print("🎯 Role Advantage Baseline System:")
    print("   1. Tracks separate baselines for spy and civilian performance")
    print("   2. EMA updates: α = 0.9 (slow adaptation)")
    print("   3. Adjusts rewards: r_final = r_raw - baseline")
    print("   4. Forces players to learn beyond current capability")
    print("   5. Multi-GPU aware: Each GPU updates independently")
    print()
    
    # Create a test generator with baseline tracking
    test_generator = CLEVRSpotDiffGenerator()
    print(f"Initial baselines: spy={test_generator.b_spy:.4f}, civ={test_generator.b_civ:.4f}")
    
    # Simulate several games to test baseline evolution
    print("\n--- Simulating Baseline Evolution ---")
    test_games = [
        # Game 1: Spy performs well, civilians perform poorly
        {"spy_r": 0.5, "civ_avg": -0.2, "name": "Spy dominates"},
        # Game 2: Spy performs poorly, civilians perform well
        {"spy_r": -0.3, "civ_avg": 0.4, "name": "Civilians catch spy"},
        # Game 3: Balanced performance
        {"spy_r": 0.1, "civ_avg": 0.0, "name": "Balanced game"},
        # Game 4: Spy adapts (performs moderately)
        {"spy_r": 0.0, "civ_avg": 0.1, "name": "Spy adapts"},
        # Game 5: Civilians adapt (better performance)
        {"spy_r": -0.1, "civ_avg": 0.3, "name": "Civilians improve"},
    ]
    
    spy_baseline_history = [test_generator.b_spy]
    civ_baseline_history = [test_generator.b_civ]
    
    for i, game in enumerate(test_games):
        print(f"\nGame {i+1}: {game['name']}")
        print(f"  Raw performance: spy_r={game['spy_r']:.3f}, civ_avg={game['civ_avg']:.3f}")
        
        # Calculate adjusted rewards
        raw_rewards = [game['civ_avg'], game['spy_r'], game['civ_avg'], game['civ_avg']]  # 4 players, player 2 is spy
        adjusted_rewards = test_generator.apply_role_advantage_adjustment(raw_rewards, spy_player=2, process_index=0)
        
        # Track baseline evolution
        spy_baseline_history.append(test_generator.b_spy)
        civ_baseline_history.append(test_generator.b_civ)
        
        print(f"  Raw rewards: {[f'{r:.3f}' for r in raw_rewards]}")
        print(f"  Adjusted rewards: {[f'{r:.3f}' for r in adjusted_rewards]}")
        print(f"  New baselines: spy={test_generator.b_spy:.4f}, civ={test_generator.b_civ:.4f}")
    
    print("\n--- Baseline Evolution Summary ---")
    print("Game |  Spy Baseline | Civ Baseline")
    print("-----|---------------|-------------")
    for i, (spy_b, civ_b) in enumerate(zip(spy_baseline_history, civ_baseline_history)):
        print(f"{i:4d} | {spy_b:12.4f} | {civ_b:11.4f}")
    
    print(f"\nBaseline adaptation:")
    print(f"  Spy baseline: {spy_baseline_history[0]:.4f} → {spy_baseline_history[-1]:.4f} (Δ={spy_baseline_history[-1]-spy_baseline_history[0]:.4f})")
    print(f"  Civ baseline: {civ_baseline_history[0]:.4f} → {civ_baseline_history[-1]:.4f} (Δ={civ_baseline_history[-1]-civ_baseline_history[0]:.4f})")
    
    # Test unified baseline system 
    print("\n--- Testing Unified Baseline System (Multi-GPU Simulation) ---")
    
    # Create a fresh generator for unified baseline testing
    unified_test_generator = CLEVRSpotDiffGenerator()
    print(f"Fresh generator baselines: spy={unified_test_generator.b_spy:.4f}, civ={unified_test_generator.b_civ:.4f}")
    
    # Simulate multi-GPU scenario with different local rewards
    print("\nSimulating 8 GPUs with different local game results:")
    
    # Simulate 8 different local game results (what each GPU sees)
    local_game_results = [
        {"spy_r": 0.4, "civ_avg": -0.1, "gpu": 0},   # GPU 0: Spy does well
        {"spy_r": -0.2, "civ_avg": 0.3, "gpu": 1},   # GPU 1: Civilians catch spy
        {"spy_r": 0.1, "civ_avg": 0.0, "gpu": 2},    # GPU 2: Balanced
        {"spy_r": 0.3, "civ_avg": -0.2, "gpu": 3},   # GPU 3: Spy dominates
        {"spy_r": -0.1, "civ_avg": 0.2, "gpu": 4},   # GPU 4: Civilians slightly better
        {"spy_r": 0.0, "civ_avg": 0.1, "gpu": 5},    # GPU 5: Close game
        {"spy_r": 0.2, "civ_avg": -0.05, "gpu": 6},  # GPU 6: Spy advantage
        {"spy_r": -0.3, "civ_avg": 0.4, "gpu": 7},   # GPU 7: Civilians dominate
    ]
    
    # Calculate what the global averages would be
    global_spy_avg = sum(result["spy_r"] for result in local_game_results) / len(local_game_results)
    global_civ_avg = sum(result["civ_avg"] for result in local_game_results) / len(local_game_results)
    
    print("\nLocal results on each GPU:")
    for result in local_game_results:
        print(f"  GPU {result['gpu']}: spy_r={result['spy_r']:+.3f}, civ_avg={result['civ_avg']:+.3f}")
    
    print(f"\nGlobal averages (what unified system would see):")
    print(f"  Global spy average: {global_spy_avg:.4f}")
    print(f"  Global civilian average: {global_civ_avg:.4f}")
    
    # Simulate unified baseline update
    old_spy_baseline = unified_test_generator.b_spy
    old_civ_baseline = unified_test_generator.b_civ
    
    unified_test_generator.update_unified_role_baselines(global_spy_avg, global_civ_avg, process_index=0)
    
    print(f"\nUnified baseline update:")
    print(f"  Spy baseline: {old_spy_baseline:.4f} → {unified_test_generator.b_spy:.4f}")
    print(f"  Civ baseline: {old_civ_baseline:.4f} → {unified_test_generator.b_civ:.4f}")
    
    # Show how different local rewards would be adjusted using the unified baseline
    print(f"\nReward adjustments using unified baseline:")
    print(f"GPU | Local Spy | Adj Spy | Local Civ | Adj Civ | Description")
    print(f"----|----------|---------|-----------|---------|-------------")
    
    for result in local_game_results:
        # Simulate reward adjustment for this GPU
        local_rewards = [result["civ_avg"], result["spy_r"], result["civ_avg"], result["civ_avg"]]  # 4 players, player 2 is spy
        
        # Apply adjustment manually (simulate what unified system would do)
        adj_spy = result["spy_r"] - unified_test_generator.b_spy
        adj_civ = result["civ_avg"] - unified_test_generator.b_civ
        
        description = ""
        if result["spy_r"] > global_spy_avg:
            description += "spy+"
        elif result["spy_r"] < global_spy_avg:
            description += "spy-"
        else:
            description += "spy="
            
        if result["civ_avg"] > global_civ_avg:
            description += "civ+"
        elif result["civ_avg"] < global_civ_avg:
            description += "civ-"
        else:
            description += "civ="
        
        print(f"{result['gpu']:3d} | {result['spy_r']:+8.3f} | {adj_spy:+7.3f} | {result['civ_avg']:+9.3f} | {adj_civ:+7.3f} | {description}")
    
    print("\n🎯 Unified Baseline Benefits:")
    print("   ✅ All GPUs use exactly same baseline (perfect synchronization)")
    print("   ✅ Prevents baseline drift between GPUs")
    print("   ✅ Uses global performance information (better than local)")
    print("   ✅ More stable training across all GPUs")
    print("   ✅ Fair comparison: all players judged by same standard")
    print()
    print("🔧 Unified Implementation Details:")
    print("   - Each GPU calculates local spy/civilian performance")
    print("   - accelerator.reduce() collects all GPU data")
    print("   - Global averages computed across all GPUs")
    print("   - Same baseline update applied to all GPUs")
    print("   - All GPUs guaranteed to have identical baselines")
    print()
    print("⚠️  Multi-GPU Advantages:")
    print("   - Unified baseline prevents GPU inconsistency")
    print("   - Better statistics (8x more data for baseline)")
    print("   - Automatic load balancing (weak/strong GPUs averaged)")
    print("   - Synchronous updates ensure fairness")
    print("   - Robust against individual GPU variance")
    
    print("\n" + "="*60)
    print("=== UNIFIED Role Advantage System Ready for Training ===")
    print("="*60)
    print("The UNIFIED role advantage baseline system has been successfully integrated!")
    print("Key features:")
    print("  🎯 Unified baselines for spy and civilian roles across ALL GPUs")
    print("  🔄 EMA updates with α=0.9 for stability")
    print("  🎮 Forces players to learn beyond current capability") 
    print("  🖥️  Multi-GPU synchronized (8 GPUs with unified baseline)")
    print("  🚫 Decision phase unchanged (only clue phase affected)")
    print("  🔒 Perfect synchronization prevents GPU baseline drift")
    print()
    print("UNIFIED Usage in training:")
    print("  1. All GPUs start with same baselines (spy=0.0, civ=0.0)")
    print("  2. Each GPU calculates local spy/civilian performance")
    print("  3. accelerator.reduce() collects data from all 8 GPUs")
    print("  4. Global averages computed and broadcast to all GPUs")
    print("  5. All GPUs update baselines with same global values")
    print("  6. Rewards adjusted: r_final = r_raw - unified_baseline")
    print("  7. All players judged by same global standard")
    print()
    print("Expected UNIFIED training dynamics:")
    print("  - Perfect baseline synchronization across all GPUs")
    print("  - Better statistics (8x more data for baseline estimation)")
    print("  - Prevents individual GPU bias or drift")
    print("  - Fair competition: same baseline standard for all")
    print("  - More stable and robust training progression")
    print("  - Automatic load balancing between strong/weak GPUs")
    print()
    print("🆕 NEW vs OLD System:")
    print("  OLD: Each GPU maintains separate baselines → potential drift")
    print("  NEW: Unified baseline across all GPUs → guaranteed consistency")
    print("  OLD: Local data only for baseline updates")
    print("  NEW: Global data from all 8 GPUs for better estimates")
    print("  Result: More reliable and fair role advantage adjustment!")