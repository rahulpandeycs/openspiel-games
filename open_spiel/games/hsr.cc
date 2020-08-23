// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/games/hsr.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace hsr {
namespace {

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"hsr",
    /*long_name=*/"HSR",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/{}  // no parameters
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new HSRGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

std::string PlayerToState(Player player) {
  switch (player) {
    case 0:
//      return CellState::kCross;
        return "Proponent";
    case 1:
        return "Opponent";
//      return CellState::kNought;
    default:
      SpielFatalError(absl::StrCat("Invalid player id ", player));
//      return CellState::kEmpty;
      return ".";
  }
}

std::string StateToString(CellState state) {

  switch (state) {
    case CellState::kEmpty:
      return ".";
    case CellState::kNought:
      return "o";
    case CellState::kCross:
      return "x";
    default:
      SpielFatalError("Unknown state.");
  }
}

void HSRState::DoApplyAction(Action move) {
  std:: cout << "\nCurrent Player: "  << (current_player_ == 1 ? "Opponent" : "Proponent") << "\n";
  std:: cout << "Current Move: "  << move << "\n";

  if(current_player_ == 1 && move < 1)
     EFG_ANDR();
  else if(current_player_ == 1 && move > 0)
     EFG_ANDL();

  if(current_player_ == 0 && (board_[move-1] == CellState::kEmpty))
     Exists(move);

//  board_[move] = PlayerToState(CurrentPlayer());

  if(PWin()){
     outcome_ = 1 - current_player_;
  }else if(OWin()) {
     outcome_ = current_player_;
  }

//  if (HasLine(current_player_)) {
//    outcome_ = current_player_;
//  }
  current_player_ = 1 - current_player_;
}

std::vector<Action> HSRState::LegalActions() const {
  if (IsTerminal()) return {};
  // Can move in any empty cell.
  std::vector<Action> moves;

  if(current_player_ == 1){
     moves.push_back(0);
     moves.push_back(1);
//     for (int cell = 1; cell < 10; cell++) {
//                  moves.push_back(cell);
//     }
  }else {
         for (int cell = 0; cell < kNumCells; ++cell) {
              if (board_[cell] == CellState::kEmpty) {
                  moves.push_back(cell+1);
              }
         }
  }
  return moves;
}

std::string HSRState::ActionToString(Player player,
                                           Action action_id) const {
  if(current_player_ == 0)
       return absl::StrCat(PlayerToState(player) , "(",
                      action_id % kNumCols, ")");
  else{
     return absl::StrCat(PlayerToState(player) , "(",
                           action_id == 0 ? "L" : "R" , ")");
  }
}

bool HSRState::PWin() const {
  return (kNumRungs == 1);
}

bool HSRState::OWin() const {
  return (kNumJars == 0 || kNumQuestions == 0 || kNumRungs == 0);
}

void HSRState::EFG_ANDR() {
     kNumQuestions = kNumQuestions-1;
     kNumRungs = kNumCells - hsr_guess;

     board_[hsr_guess-1] = CellState::kNought;

     int start = hsr_guess - 1;
     while(start > 0){
           board_[start-1] = CellState::kCross;
           start--;
     }
}

void HSRState::EFG_ANDL() {

     for(int start = hsr_guess; start < kNumCells + 1; start++){
         board_[start-1] = CellState::kCross;
     }

     kNumQuestions--;
     kNumRungs = hsr_guess;
     kNumJars--;
}

void HSRState::Exists(Action move) {
     hsr_guess = move;
}

//bool HSRState::IsFull() const { return num_moves_ == kNumCells; }

HSRState::HSRState(std::shared_ptr<const Game> game) : State(game) {
  std::fill(begin(board_), end(board_), CellState::kEmpty);
}

std::string HSRState::ToString() const {
  std::string str;
  for (int r = 0; r < kNumRows; ++r) {
    for (int c = 0; c < kNumCols; ++c) {
      absl::StrAppend(&str, StateToString(BoardAt(r, c)));
    }
    if (r < (kNumRows - 1)) {
      absl::StrAppend(&str, "\n");
    }
  }
  return str;
}

bool HSRState::IsTerminal() const {
  return outcome_ != kInvalidPlayer;
}

std::vector<double> HSRState::Returns() const {
  if (outcome_ == Player{0}) {
    return {1.0, -1.0};
  } else if (outcome_ == Player{1}) {
    return {-1.0, 1.0};
  } else {
    return {0.0, 0.0};
  }
}

std::string HSRState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string HSRState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void HSRState::ObservationTensor(Player player,
                                       std::vector<double>* values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  // Treat `values` as a 2-d tensor.
  TensorView<2> view(values, {kCellStates, kNumCells}, true);
  for (int cell = 0; cell < kNumCells; ++cell) {
    view[{static_cast<int>(board_[cell]), cell}] = 1.0;
  }
}

void HSRState::UndoAction(Player player, Action move) {
  board_[move] = CellState::kEmpty;
  current_player_ = player;
  outcome_ = kInvalidPlayer;
//  num_moves_ -= 1;
  history_.pop_back();
}

std::unique_ptr<State> HSRState::Clone() const {
  return std::unique_ptr<State>(new HSRState(*this));
}

HSRGame::HSRGame(const GameParameters& params)
    : Game(kGameType, params) {}

}  // namespace HSR
}  // namespace open_spiel
