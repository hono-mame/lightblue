{-# LANGUAGE OverloadedStrings, RecordWildCards #-}
module DTS.Prover.Oracle.Oracle
    ( OracleMode(..)
    , OracleConfig(..)
    , defaultOracleConfig
    , oracle
    , oracleBuilder
    , oracleBuilderWithConfig
    ) where

import Control.Monad (unless)
import qualified DTS.DTTdeBruijn as DdB
import qualified Data.Map.Strict as M
import qualified Data.Text.Lazy as T
import System.Directory (doesFileExist)
import System.Environment (lookupEnv)
import System.Exit (die)
import System.FilePath ((</>))
import DTS.Prover.Oracle.OracleUtils
    ( loadWordEmbeddings
    , calcPScore
    , loadMlpWeights
    , calcMlpScore
    )

data OracleMode
    = OracleHEC
    | OracleOovHEC
    | OracleOovMLP
    | OracleStandaloneHEC
    | OracleStandaloneMLP
    deriving (Show, Eq)

data OracleConfig = OracleConfig
    { ocMode          :: OracleMode
    , ocEmbeddingPath :: Maybe FilePath
    , ocOovEmbPath    :: Maybe FilePath
    , ocMlpWeightPath :: Maybe FilePath
    , ocThreshold     :: Float
    } deriving (Show, Eq)

defaultOracleConfig :: OracleConfig
defaultOracleConfig = OracleConfig
    { ocMode          = OracleHEC
    , ocEmbeddingPath = Nothing
    , ocOovEmbPath    = Nothing
    , ocMlpWeightPath = Nothing
    , ocThreshold     = 0.5
    }

-- | Calibrate probability score s relative to threshold th so that (calibrateScore th s > 0.5) <=> (s > th).
-- When th == 0.5, returns s unchanged.
calibrateScore :: Float -> Float -> Float
calibrateScore th s
    | abs (th - 0.5) < 1e-6 = s
    | s <= 0.0              = 0.0
    | s >= 1.0              = 1.0
    | otherwise             =
        let num = s * (1.0 - th)
            den = num + (1.0 - s) * th
        in if den <= 0.0 then 0.0 else num / den

resolveDataFile :: String -> Maybe FilePath -> IO FilePath
resolveDataFile _ (Just path) = return path
resolveDataFile defaultFileName Nothing = do
    mLightblue <- lookupEnv "LIGHTBLUE"
    case mLightblue of
        Just lb | not (null lb) -> return (lb </> "data" </> defaultFileName)
        _ -> die $ unlines
            [ "LIGHTBLUE environment variable is not set."
            , "Set LIGHTBLUE to the lightblue directory or specify explicit file paths for " ++ defaultFileName ++ "."
            ]

ensureFileExists :: String -> FilePath -> IO ()
ensureFileExists label path = do
    exists <- doesFileExist path
    unless exists $ die $ unlines
        [ label ++ " file not found: " ++ path
        ]

scorePairMap :: ( [Float] -> [Float] -> Float )
             -> M.Map T.Text [[Float]]
             -> T.Text
             -> T.Text
             -> Maybe Float
scorePairMap scorer embMap childKey parentKey =
    case (M.lookup parentKey embMap, M.lookup childKey embMap) of
        (Just pEmbs, Just cEmbs) ->
            let allScores = [ scorer pEmb cEmb | pEmb <- pEmbs, cEmb <- cEmbs ]
            in if null allScores
               then Nothing
               else Just (maximum allScores)
        _ -> Nothing

oracle :: (DdB.ConName, DdB.ConName) -> IO Float
oracle (childName, parentName) = do
    oracleFun <- oracleBuilder Nothing
    return $ oracleFun childName parentName

oracleBuilder :: Maybe FilePath -> IO (DdB.ConName -> DdB.ConName -> Float)
oracleBuilder mEmbPath =
    oracleBuilderWithConfig defaultOracleConfig { ocEmbeddingPath = mEmbPath }

oracleBuilderWithConfig :: OracleConfig -> IO (DdB.ConName -> DdB.ConName -> Float)
oracleBuilderWithConfig OracleConfig{..} = do
    let alpha = 0.1
        stripTilde = T.filter (/= '~')
    case ocMode of
        OracleHEC -> do
            embPath <- resolveDataFile "Embedding.csv" ocEmbeddingPath
            ensureFileExists "Oracle embedding" embPath
            embMap <- loadWordEmbeddings embPath
            return $ \childName parentName ->
                let cKey = stripTilde childName
                    pKey = stripTilde parentName
                in maybe 0.0 (calibrateScore ocThreshold) (scorePairMap (calcPScore alpha) embMap cKey pKey)

        OracleOovHEC -> do
            embPath <- resolveDataFile "Embedding.csv" ocEmbeddingPath
            oovPath <- resolveDataFile "Embedding_OOV_HEC.csv" ocOovEmbPath
            ensureFileExists "Oracle base embedding" embPath
            ensureFileExists "Oracle OOV HEC embedding" oovPath
            baseMap <- loadWordEmbeddings embPath
            oovMap  <- loadWordEmbeddings oovPath
            return $ \childName parentName ->
                let cKey = stripTilde childName
                    pKey = stripTilde parentName
                in case scorePairMap (calcPScore alpha) baseMap cKey pKey of
                    Just s  -> s
                    Nothing -> maybe 0.0 (calibrateScore ocThreshold) (scorePairMap (calcPScore alpha) oovMap cKey pKey)

        OracleOovMLP -> do
            embPath    <- resolveDataFile "Embedding.csv" ocEmbeddingPath
            oovPath    <- resolveDataFile "Embedding_OOV_MLP.csv" ocOovEmbPath
            weightPath <- resolveDataFile "MLP_Head_Weights.csv" ocMlpWeightPath
            ensureFileExists "Oracle base embedding" embPath
            ensureFileExists "Oracle OOV MLP embedding" oovPath
            ensureFileExists "Oracle MLP head weights" weightPath
            baseMap <- loadWordEmbeddings embPath
            oovMap  <- loadWordEmbeddings oovPath
            weights <- loadMlpWeights weightPath
            return $ \childName parentName ->
                let cKey = stripTilde childName
                    pKey = stripTilde parentName
                in case scorePairMap (calcPScore alpha) baseMap cKey pKey of
                    Just s  -> s
                    Nothing -> maybe 0.0 (calibrateScore ocThreshold) (scorePairMap (calcMlpScore weights) oovMap cKey pKey)

        OracleStandaloneHEC -> do
            oovPath <- resolveDataFile "Embedding_OOV_HEC.csv" (case ocOovEmbPath of Just p -> Just p; Nothing -> ocEmbeddingPath)
            ensureFileExists "Oracle standalone HEC embedding" oovPath
            oovMap <- loadWordEmbeddings oovPath
            return $ \childName parentName ->
                let cKey = stripTilde childName
                    pKey = stripTilde parentName
                in maybe 0.0 (calibrateScore ocThreshold) (scorePairMap (calcPScore alpha) oovMap cKey pKey)

        OracleStandaloneMLP -> do
            oovPath    <- resolveDataFile "Embedding_OOV_MLP.csv" (case ocOovEmbPath of Just p -> Just p; Nothing -> ocEmbeddingPath)
            weightPath <- resolveDataFile "MLP_Head_Weights.csv" ocMlpWeightPath
            ensureFileExists "Oracle standalone MLP embedding" oovPath
            ensureFileExists "Oracle MLP head weights" weightPath
            oovMap  <- loadWordEmbeddings oovPath
            weights <- loadMlpWeights weightPath
            return $ \childName parentName ->
                let cKey = stripTilde childName
                    pKey = stripTilde parentName
                in maybe 0.0 (calibrateScore ocThreshold) (scorePairMap (calcMlpScore weights) oovMap cKey pKey)
