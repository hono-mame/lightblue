{-# LANGUAGE OverloadedStrings, RecordWildCards #-}

module DTS.Prover.Oracle.OracleUtils
    ( loadWordEmbeddings
    , calcPScore
    , MlpWeights(..)
    , loadMlpWeights
    , calcMlpLogit
    , calcMlpScore
    ) where

import Data.List (zipWith)
import qualified Data.Map.Strict as M
import qualified Data.Text.Lazy as T
import qualified Data.Text.Lazy.IO as TIO
import qualified Data.Text.Lazy.Read as TR
import Data.Maybe (mapMaybe)

eps :: Float
eps = 1e-6

norm :: [Float] -> Float
norm xs = sqrt $ sum $ map (\x -> x*x) xs

dot :: [Float] -> [Float] -> Float
dot xs ys = sum $ zipWith (*) xs ys

clip :: Float -> Float
clip x = max (-1 + eps) (min (1 - eps) x)

sigmoid :: Float -> Float
sigmoid x = 1 / (1 + exp x)

angleChild :: Float -> [Float] -> [Float] -> Float
angleChild _k parent child =
    let
        normParent = norm parent
        normParentSq = normParent * normParent
        normChild = norm child
        normChildSq = normChild * normChild
        euclidDist = max (norm $ zipWith (-) parent child) eps
        dotProd = dot parent child
        g = 1 + normParentSq * normChildSq - 2 * dotProd
        gSqrt = sqrt g
        childNumerator = dotProd * (1 + normParentSq) - normParentSq * (1 + normChildSq)
        childDenominator = euclidDist * normParent * gSqrt
        cosAngleChild = clip (childNumerator / childDenominator)
    in acos cosAngleChild

angleParent :: Float -> [Float] -> [Float] -> Float
angleParent k parent _child =
    let
        normParent = norm parent
    in asin (k * (1 - normParent * normParent) / normParent)

loadWordEmbeddings :: FilePath -> IO (M.Map T.Text [[Float]])
loadWordEmbeddings path = do
    contents <- TIO.readFile path
    let ls = drop 1 $ T.lines contents
    let parseFloat t = case TR.double (T.strip t) of
              Right (d, _) -> Just (realToFrac d :: Float)
              _ -> Nothing
    let parseLine line = case T.splitOn "," (T.strip line) of
              (wordTxt : dimsTxt) | not (T.null wordTxt) -> Just (wordTxt, mapMaybe parseFloat dimsTxt)
              _ -> Nothing
    let parsedLines = mapMaybe parseLine ls
    pure $ M.fromListWith (++) [(word, [dims]) | (word, dims) <- parsedLines]

calcPScore :: Float -> [Float] -> [Float] -> Float
calcPScore alpha pEmb cEmb =
    let a1 = angleChild alpha pEmb cEmb
        a2 = angleParent alpha pEmb cEmb
        score = a1 - a2
    in sigmoid score

-- | Weights for PairMLP classifier head (80 -> 32 -> 1)
data MlpWeights = MlpWeights
    { mlpW1 :: [[Float]] -- ^ Layer 1 weight matrix (32 x 80)
    , mlpB1 :: [Float]   -- ^ Layer 1 bias vector (32)
    , mlpW2 :: [Float]   -- ^ Layer 2 weight vector (32)
    , mlpB2 :: Float     -- ^ Layer 2 scalar bias (1)
    } deriving (Show, Eq)

loadMlpWeights :: FilePath -> IO MlpWeights
loadMlpWeights path = do
    contents <- TIO.readFile path
    let ls = filter (not . T.null) $ map T.strip $ T.lines contents
        parseFloat t = case TR.double (T.strip t) of
            Right (d, _) -> Just (realToFrac d :: Float)
            _ -> Nothing
        parseRow tag line = case T.splitOn "," line of
            (t : _idx : vals) | T.strip t == tag -> Just (mapMaybe parseFloat vals)
            _ -> Nothing
        w1Rows = mapMaybe (parseRow "w1") ls
        b1Row  = concat $ mapMaybe (parseRow "b1") ls
        w2Row  = concat $ mapMaybe (parseRow "w2") ls
        b2Vals = concat $ mapMaybe (parseRow "b2") ls
        b2Val  = case b2Vals of
            (v : _) -> v
            []      -> 0.0
    pure $ MlpWeights w1Rows b1Row w2Row b2Val

-- | Compute raw logit of PairMLP from 20-dim parent (hypernym) and child (hyponym) features.
-- Feature vector is [a, b, |a - b|, a * b] where a = child (hypo), b = parent (hyper).
calcMlpLogit :: MlpWeights -> [Float] -> [Float] -> Float
calcMlpLogit MlpWeights{..} pEmb cEmb =
    let a = cEmb
        b = pEmb
        diffAbs = zipWith (\x y -> abs (x - y)) a b
        prod = zipWith (*) a b
        feat = a ++ b ++ diffAbs ++ prod
        hidden = zipWith (\row bias -> max 0.0 (bias + dot row feat)) mlpW1 mlpB1
    in mlpB2 + dot mlpW2 hidden

-- | Compute logistic sigmoid probability P(child <= parent) from PairMLP features.
calcMlpScore :: MlpWeights -> [Float] -> [Float] -> Float
calcMlpScore weights pEmb cEmb =
    let logit = calcMlpLogit weights pEmb cEmb
    in 1.0 / (1.0 + exp (-logit))
